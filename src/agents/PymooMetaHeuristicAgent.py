# src/agents/PymooMetaHeuristicAgent.py
import numpy as np
import time
import sys
import os
import json
import random

# Pymoo imports
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.crossover.pntx import SinglePointCrossover
    from pymoo.operators.mutation.pm import PM, PolynomialMutation
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
except ImportError:
    print("FATAL ERROR: pymoo is not installed. Please install it using 'pip install pymoo'", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

try:
    from src.utils.simulation_logger import SimulationLogger
except ImportError:
    print("Warning: SimulationLogger utility not found. Detailed step logging will be disabled.")
    SimulationLogger = None

# Heuristic IDs (can be defined as constants)
HEURISTIC_COP = 0
HEURISTIC_BSP = 1
HEURISTIC_BSPEW = 2

# --- Pymoo Problem Definition ---
class InventoryOptimizationProblem(Problem):
    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.n_items = agent_instance.n_items

        n_vars = self.n_items * 3
        
        xl, xu = [], []
        for i in range(self.n_items):
            epsilon = 1e-9 # A small number to add to the upper bound if xl == xu
            valid_suppliers = np.where(self.agent.item_supplier_matrix[i, :] == 1)[0]
            _xl_supp = 0.0
            _xu_supp = float(len(valid_suppliers) - 1 if len(valid_suppliers) > 0 else 0)
            if _xl_supp >= _xu_supp:
                _xu_supp = _xl_supp + epsilon
            xl.append(_xl_supp)
            xu.append(_xu_supp)

            # Variable 2: Heuristic ID
            _xl_heur = 0.0
            _xu_heur = 2.0
            if _xl_heur >= _xu_heur: # Unlikely here, but good practice
                 _xu_heur = _xl_heur + epsilon
            xl.append(_xl_heur)
            xu.append(_xu_heur)

            # Variable 3: Parameter Index
            max_param_len = max(len(self.agent.quantity_options), len(self.agent.base_stock_level_options))
            _xl_param = 0.0
            _xu_param = float(max_param_len - 1 if max_param_len > 0 else 0)
            if _xl_param >= _xu_param:
                _xu_param = _xl_param + epsilon
            xl.append(_xl_param)
            xu.append(_xu_param)
        
        super().__init__(n_var=n_vars, n_obj=1, n_constr=0, xl=np.array(xl), xu=np.array(xu))

    def _evaluate(self, X, out, *args, **kwargs):
        fitness_values = []
        for i in range(X.shape[0]):
            x_individual = X[i, :]
            chromosome = self._decode_individual(x_individual)
            total_reward_across_episodes = 0.0
            if self.agent.num_optimize_eval_episodes == 0:
                fitness = -np.inf
            else:
                for _ in range(self.agent.num_optimize_eval_episodes):
                    self.agent.env.reset()
                    terminated, truncated = False, False
                    episode_reward = 0.0
                    while not (terminated or truncated):
                        action = self.agent._get_action_from_chromosome(chromosome)
                        _, reward, terminated, truncated, _ = self.agent.env.step(action, verbose=False)
                        episode_reward += reward
                    total_reward_across_episodes += episode_reward
                fitness = total_reward_across_episodes / self.agent.num_optimize_eval_episodes
            fitness_values.append(-fitness)
        out["F"] = np.array(fitness_values)

    def _decode_individual(self, x_individual):
        chromosome = []
        for i in range(self.n_items):
            base_idx = i * 3
            
            valid_suppliers = np.where(self.agent.item_supplier_matrix[i, :] == 1)[0]
            if not valid_suppliers.size:
                supplier_idx = 0
            else:
                valid_supplier_list_idx = int(round(x_individual[base_idx]))
                supplier_idx = valid_suppliers[valid_supplier_list_idx]
            
            heuristic_id = int(round(x_individual[base_idx + 1]))
            
            param_idx = int(round(x_individual[base_idx + 2]))
            param_value = 0.0
            if heuristic_id == HEURISTIC_COP:
                options = self.agent.quantity_options
                if options:
                    actual_idx = min(param_idx, len(options) - 1)
                    param_value = options[actual_idx]
            else:
                options = self.agent.base_stock_level_options
                if options:
                    actual_idx = min(param_idx, len(options) - 1)
                    param_value = options[actual_idx]
            
            chromosome.append((int(supplier_idx), int(heuristic_id), float(param_value)))
        return chromosome


class PymooMetaHeuristicAgent:
    def __init__(self, env,
                 algorithm_config: dict,
                 num_optimize_eval_episodes: int = 10,
                 num_final_eval_episodes: int = 50,
                 quantity_options: list = None,
                 base_stock_level_options: list = None,
                 bspew_waste_estimation_method: str = "deterministic_simulation",
                 bspew_waste_horizon_review_periods: int = 1,
                 bspew_num_ew_demand_sim_paths: int = 30,
                 load_policy_path: str = None,
                 save_policy_path: str = None,
                 logger_settings: dict = None,
                 **other_kwargs):

        self.env = env
        self.n_items = env.n_items
        self.n_suppliers = env.n_suppliers
        self.item_supplier_matrix = env.item_supplier_matrix
        self.algorithm_config = algorithm_config
        self.num_optimize_eval_episodes = num_optimize_eval_episodes
        self.num_final_eval_episodes = num_final_eval_episodes
        self.quantity_options = quantity_options if quantity_options is not None else [0, 1, 5, 10]
        self.base_stock_level_options = base_stock_level_options if base_stock_level_options is not None else [0, 5, 10, 20, 50]
        if not self.quantity_options: print("Warning: PymooMetaHeuristicAgent quantity_options is empty.")
        if not self.base_stock_level_options: print("Warning: PymooMetaHeuristicAgent base_stock_level_options is empty.")
        self.bspew_waste_estimation_method = bspew_waste_estimation_method
        self.bspew_waste_horizon_review_periods = bspew_waste_horizon_review_periods
        self.bspew_num_ew_demand_sim_paths = bspew_num_ew_demand_sim_paths
        ew_rng_seed_offset = 78901
        base_seed_for_ew = self.env._initial_seed if hasattr(self.env, '_initial_seed') and self.env._initial_seed is not None else random.randint(0, 1e9-ew_rng_seed_offset)
        self.ew_sim_rng = np.random.default_rng(base_seed_for_ew + ew_rng_seed_offset)
        self.logger = None
        if SimulationLogger and logger_settings and logger_settings.get("log_step_details", False):
            exp_name_for_logger = logger_settings.get("experiment_name", f"pymoo_meta_{self.algorithm_config.get('name', 'default')}_experiment")
            self.logger = SimulationLogger(
                log_dir=logger_settings.get("log_dir", "./src/results/simulation_logs"),
                experiment_name=exp_name_for_logger,
                log_step_details=logger_settings.get("log_step_details", True),
                log_actions=logger_settings.get("log_actions", True),
                n_items=self.env.n_items, n_suppliers=self.env.n_suppliers
            )
            if self.logger.log_step_details_enabled:
                print(f"Pymoo Agent: Detailed simulation logging enabled. Log file: {self.logger.log_file_path}")
        self.best_chromosome = None
        self.load_policy_path = load_policy_path
        self.save_policy_path = save_policy_path
        if self.load_policy_path:
            print(f"\n--- Loading Pre-optimized Meta-Heuristic Policy ---")
            try:
                with open(self.load_policy_path, 'r') as f: self.best_chromosome = json.load(f)
                print(f"Policy successfully loaded from: {self.load_policy_path}")
                if not isinstance(self.best_chromosome, list) or len(self.best_chromosome) != self.n_items:
                    raise ValueError("Loaded policy has incorrect format or length.")
                print(f"Loaded Policy (Chromosome):\n{self.best_chromosome}")
            except Exception as e:
                print(f"Error loading policy from '{self.load_policy_path}': {e}. Will optimize.", file=sys.stderr)
                self.load_policy_path = None
                self.best_chromosome = None
        if self.best_chromosome is None:
            self._optimize_policy_pymoo()
            if self.save_policy_path:
                print(f"\n--- Saving Optimized Meta-Heuristic Policy ---")
                try:
                    os.makedirs(os.path.dirname(self.save_policy_path), exist_ok=True)
                    with open(self.save_policy_path, 'w') as f: json.dump(self.best_chromosome, f, indent=4)
                    print(f"Optimized policy saved to: {self.save_policy_path}")
                except Exception as e:
                    print(f"Error saving policy to '{self.save_policy_path}': {e}", file=sys.stderr)

    def _optimize_policy_pymoo(self):
        algo_name = self.algorithm_config.get("name", "GA").upper()
        algo_params = self.algorithm_config.get("params", {})
        print(f"--- Optimizing Meta-Heuristic Policy with Pymoo ({algo_name}) ---")
        print(f"  - Algorithm Params: {algo_params}")
        print(f"  - Num Eval Episodes/Individual (Optimization): {self.num_optimize_eval_episodes}")
        problem = InventoryOptimizationProblem(self)
        pop_size = algo_params.get("pop_size", 50)
        if algo_name == "GA":
            algorithm = GA(
                pop_size=pop_size,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=algo_params.get("crossover_rate", 0.9), eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True
            )
        elif algo_name == "NSGA2":
            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=algo_params.get("crossover_rate", 0.9), eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True
            )
        elif algo_name == "PSO":
            algorithm = PSO(pop_size=pop_size)
        else:
            raise ValueError(f"Unknown algorithm '{algo_name}' specified in algorithm_config.")
        termination_config = algo_params.get("termination", {"n_gen": 100})
        print(f"  - Termination Criteria: {termination_config}")
        termination_args = []
        for key, value in termination_config.items():
            termination_args.append(key)
            termination_args.append(value)
        termination = get_termination(*termination_args)
        start_time = time.time()
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=self.env._initial_seed if hasattr(self.env, '_initial_seed') and self.env._initial_seed is not None else None,
            verbose=True,
            save_history=False
        )
        end_time = time.time()
        print(f"\nPymoo ({algo_name}) Optimization finished in {end_time - start_time:.2f} seconds.")
        if res.X is not None:
            best_fitness = -res.F[0]
            best_solution_pymoo = res.X
            self.best_chromosome = problem._decode_individual(best_solution_pymoo)
            print(f"Optimized Policy (Best Chromosome) Found:\n{self.best_chromosome}")
            print(f"Best Fitness found (avg reward): {best_fitness:.2f}")
        else:
            print("Pymoo Error: Optimization did not return a solution. Using a default.", file=sys.stderr)
            default_gene = (0, HEURISTIC_COP, 0.0)
            self.best_chromosome = [default_gene for _ in range(self.n_items)]

    def _get_action_from_chromosome(self, chromosome):
        action_matrix = np.zeros((self.n_items, self.n_suppliers), dtype=np.float32)
        current_outstanding_orders_all_items = self._calculate_outstanding_orders()
        for i in range(self.n_items):
            if i >= len(chromosome): continue
            supplier_idx, heuristic_id, param_value = chromosome[i]
            order_qty = 0.0
            if heuristic_id == HEURISTIC_COP:
                order_qty = param_value
            elif heuristic_id == HEURISTIC_BSP or heuristic_id == HEURISTIC_BSPEW:
                inv_pos = self.env.inventory_level[i] + current_outstanding_orders_all_items[i]
                if heuristic_id == HEURISTIC_BSP:
                    order_qty = param_value - inv_pos
                else: # HEURISTIC_BSPEW
                    chosen_supplier_lead_time = int(self.env.lead_times[i, supplier_idx])
                    ew_item_i = self._calculate_ew_deterministic_simulation(i, chosen_supplier_lead_time)
                    order_qty = param_value - inv_pos + ew_item_i
            action_matrix[i, supplier_idx] = float(max(0.0, order_qty))
        return action_matrix

    def run(self, render_steps=False, verbose=False):
        all_episode_rewards = []
        if self.best_chromosome is None:
             print("Error: Agent has no optimized policy to run. Exiting.", file=sys.stderr)
             return []
        print(f"\nRunning final evaluation with {'Loaded' if self.load_policy_path else 'Optimized'} Policy "
              f"for {self.num_final_eval_episodes} episode(s)...")
        for episode_idx in range(self.num_final_eval_episodes):
            if self.logger: self.logger.start_episode(episode_num=episode_idx)
            self.env.reset()
            terminated, truncated = False, False
            total_reward_episode = 0.0
            while not (terminated or truncated):
                action_to_take = self._get_action_from_chromosome(self.best_chromosome)
                _, reward_step, terminated, truncated, info_step = self.env.step(action_to_take, verbose=verbose)
                if self.logger: self.logger.log_step(step_num=self.env.current_step, reward=reward_step, info=info_step, action=action_to_take if self.logger.log_actions else None)
                if render_steps: self.env.render()
                total_reward_episode += reward_step
            if self.logger: self.logger.end_episode()
            if verbose: print(f"Evaluation Episode {episode_idx + 1}: Total Reward: {total_reward_episode:.2f}")
            all_episode_rewards.append(total_reward_episode)
        if self.logger: self.logger.finalize_logs()
        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(f"Average reward over {self.num_final_eval_episodes} final evaluation episodes: {avg_final_reward:.2f}")
        return all_episode_rewards

    def _calculate_outstanding_orders(self) -> np.ndarray:
        outstanding = np.zeros(self.env.n_items, dtype=int)
        current_time = self.env.current_step
        lead_times = self.env.lead_times
        if hasattr(self.env, 'order_history') and self.env.order_history is not None:
            for t_placed, i_item, s_supplier, qty_ordered in self.env.order_history:
                if i_item < self.n_items and s_supplier < self.n_suppliers:
                    arrival_time = t_placed + lead_times[i_item, s_supplier]
                    if arrival_time > current_time:
                        outstanding[i_item] += qty_ordered
        return outstanding

    def _estimate_future_daily_demands_mc(self, item_idx: int, horizon: int) -> np.ndarray:
        if horizon <= 0 or self.bspew_num_ew_demand_sim_paths <= 0:
            return np.zeros(horizon)
        all_simulated_demands_for_item = np.zeros((self.bspew_num_ew_demand_sim_paths, horizon))
        original_stoch_model_rng = None
        stoch_model_had_rng_attr = hasattr(self.env.stoch_model, 'rng')
        if stoch_model_had_rng_attr:
            original_stoch_model_rng = self.env.stoch_model.rng
            self.env.stoch_model.rng = self.ew_sim_rng
        for i_path in range(self.bspew_num_ew_demand_sim_paths):
            try:
                full_scenario = self.env.stoch_model.generate_scenario(n_time_steps=horizon)
                if full_scenario.ndim == 2 and full_scenario.shape[0] == self.env.n_items:
                    all_simulated_demands_for_item[i_path, :] = full_scenario[item_idx, :horizon]
                elif full_scenario.ndim == 1 and self.env.n_items == 1:
                    all_simulated_demands_for_item[i_path, :] = full_scenario[:horizon]
                else:
                    if hasattr(self.env.stoch_model, 'generate_demands_for_item'):
                         all_simulated_demands_for_item[i_path, :] = self.env.stoch_model.generate_demands_for_item(item_idx, horizon)
                    else:
                        print(f"Warning: stoch_model.generate_scenario returned unexpected shape {full_scenario.shape}. Using zeros.", file=sys.stderr)
            except Exception as e:
                print(f"Error during stoch_model.generate_scenario for item {item_idx}: {e}. Using zeros.", file=sys.stderr)
        if stoch_model_had_rng_attr and original_stoch_model_rng is not None:
            self.env.stoch_model.rng = original_stoch_model_rng
        avg_daily_demands = np.mean(all_simulated_demands_for_item, axis=0)
        return avg_daily_demands

    def _calculate_ew_deterministic_simulation(self, item_idx: int, chosen_supplier_lead_time: int) -> float:
        current_inventory_sim = self.env.inventory_age[item_idx, :].astype(float)
        max_age_M = self.env.max_age
        sim_horizon = self.bspew_waste_horizon_review_periods + chosen_supplier_lead_time - 1
        if sim_horizon <= 0: return 0.0
        future_daily_demands_est = self._estimate_future_daily_demands_mc(item_idx, sim_horizon)
        total_simulated_waste_float = 0.0
        for t_sim in range(sim_horizon):
            demand_for_this_sim_day_float = future_daily_demands_est[t_sim]
            if demand_for_this_sim_day_float < 0: demand_for_this_sim_day_float = 0.0
            inventory_at_start_of_aging_sim = current_inventory_sim.copy()
            next_step_inventory_sim = np.zeros_like(current_inventory_sim, dtype=float)
            simulated_waste_this_step_float = 0.0
            for age_idx_sim in range(max_age_M):
                n_items_in_bin_sim_float = inventory_at_start_of_aging_sim[age_idx_sim]
                if n_items_in_bin_sim_float <= 1e-9: continue
                cdf_age_plus_1 = self.env.shelf_life_cdf[item_idx, age_idx_sim + 1] if (age_idx_sim + 1) < max_age_M else 1.0
                cdf_age = self.env.shelf_life_cdf[item_idx, age_idx_sim]
                prob_survival_up_to_age = 1.0 - cdf_age
                prob_expire_in_next_step_sim = 0.0
                if prob_survival_up_to_age > 1e-9:
                    prob_expire_in_next_step_sim = np.clip((cdf_age_plus_1 - cdf_age) / prob_survival_up_to_age, 0.0, 1.0)
                elif cdf_age_plus_1 > cdf_age:
                    prob_expire_in_next_step_sim = 1.0
                wasted_count_sim_float = n_items_in_bin_sim_float * prob_expire_in_next_step_sim
                simulated_waste_this_step_float += wasted_count_sim_float
                survivors_count_sim_float = n_items_in_bin_sim_float - wasted_count_sim_float
                if age_idx_sim < max_age_M - 1 and survivors_count_sim_float > 1e-9:
                    next_step_inventory_sim[age_idx_sim + 1] += survivors_count_sim_float
            current_inventory_sim = next_step_inventory_sim
            total_simulated_waste_float += simulated_waste_this_step_float
            demand_to_satisfy_sim_float = demand_for_this_sim_day_float
            for age_idx_sim in range(max_age_M - 1, -1, -1):
                if demand_to_satisfy_sim_float <= 1e-9: break
                available_in_bin_sim_float = current_inventory_sim[age_idx_sim]
                if available_in_bin_sim_float <= 1e-9: continue
                fulfilled_from_bin_sim_float = min(available_in_bin_sim_float, demand_to_satisfy_sim_float)
                current_inventory_sim[age_idx_sim] -= fulfilled_from_bin_sim_float
                demand_to_satisfy_sim_float -= fulfilled_from_bin_sim_float
        return float(total_simulated_waste_float)