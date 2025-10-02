# src/agents/GAMetaHeuristicAgent.py
import numpy as np
import time
import sys
import os
import json # For saving/loading policies
import random # For GA operations

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

try:
    from src.utils.simulation_logger import SimulationLogger
except ImportError:
    print("Warning: SimulationLogger utility not found. Detailed step logging will be disabled for GAMetaHeuristicAgent.")
    SimulationLogger = None

# Heuristic IDs (can be defined as constants)
HEURISTIC_COP = 0
HEURISTIC_BSP = 1
HEURISTIC_BSPEW = 2

class GAMetaHeuristicAgent:
    def __init__(self, env,
                 population_size: int = 50,
                 num_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 5,
                 num_optimize_eval_episodes: int = 10,
                 num_final_eval_episodes: int = 50,
                 quantity_options: list = None, # Should be provided
                 base_stock_level_options: list = None, # Should be provided
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

        # GA parameters
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        self.num_optimize_eval_episodes = num_optimize_eval_episodes
        self.num_final_eval_episodes = num_final_eval_episodes

        self.quantity_options = quantity_options if quantity_options is not None else [0, 1, 5, 10]
        self.base_stock_level_options = base_stock_level_options if base_stock_level_options is not None else [0, 5, 10, 20, 50]
        
        if not self.quantity_options:
            print("Warning: GAMetaHeuristicAgent quantity_options is empty or None. COP may not work well.")
        if not self.base_stock_level_options:
            print("Warning: GAMetaHeuristicAgent base_stock_level_options is empty or None. BSP/BSPEW may not work well.")


        # BSPEW fixed parameters
        self.bspew_waste_estimation_method = bspew_waste_estimation_method
        self.bspew_waste_horizon_review_periods = bspew_waste_horizon_review_periods
        self.bspew_num_ew_demand_sim_paths = bspew_num_ew_demand_sim_paths

        # RNGs
        # Use env's RNG if available for GA's main RNG, otherwise create a new one.
        # The seed for this RNG should be managed by the runner's set_seed for reproducibility of GA.
        self.ga_rng = np.random.default_rng(self.env._initial_seed if hasattr(self.env, '_initial_seed') and self.env._initial_seed is not None else random.randint(0, 1e9))
        
        # EW sim RNG, seeded independently to ensure EW calculations are consistent across evaluations
        # if they use the same underlying parameters for EW calculation.
        ew_rng_seed_offset = 78901 # Arbitrary offset for EW RNG seed
        base_seed_for_ew = self.env._initial_seed if hasattr(self.env, '_initial_seed') and self.env._initial_seed is not None else random.randint(0, 1e9-ew_rng_seed_offset)
        self.ew_sim_rng = np.random.default_rng(base_seed_for_ew + ew_rng_seed_offset)


        self.logger = None
        if SimulationLogger and logger_settings and logger_settings.get("log_step_details", False):
            exp_name_for_logger = logger_settings.get("experiment_name", "ga_meta_heuristic_default_experiment")
            self.logger = SimulationLogger(
                log_dir=logger_settings.get("log_dir", "./src/results/simulation_logs"),
                experiment_name=exp_name_for_logger,
                log_step_details=logger_settings.get("log_step_details", True),
                log_actions=logger_settings.get("log_actions", True),
                n_items=self.env.n_items,
                n_suppliers=self.env.n_suppliers
            )
            if self.logger.log_step_details_enabled:
                 print(f"GA Meta-Heuristic Agent: Detailed simulation logging enabled. Log file: {self.logger.log_file_path}")

        self.best_chromosome = None
        self.load_policy_path = load_policy_path
        self.save_policy_path = save_policy_path

        if self.load_policy_path:
            print(f"\n--- Loading Pre-optimized GA Meta-Heuristic Policy ---")
            try:
                with open(self.load_policy_path, 'r') as f:
                    self.best_chromosome = json.load(f)
                print(f"Policy successfully loaded from: {self.load_policy_path}")
                # Basic validation
                if not isinstance(self.best_chromosome, list) or len(self.best_chromosome) != self.n_items:
                    raise ValueError("Loaded policy has incorrect format or length.")
                print(f"Loaded Policy (Chromosome):\n{self.best_chromosome}")
            except Exception as e:
                print(f"Error loading GA policy from '{self.load_policy_path}': {e}. Will optimize.", file=sys.stderr)
                self.load_policy_path = None # Ensure optimization runs
                self.best_chromosome = None # Clear partially loaded
        
        if self.best_chromosome is None: # If not loaded or loading failed
            print("Optimizing GA Meta-Heuristic Policy...")
            print(f"  - Population Size: {self.population_size}")
            print(f"  - Num Generations: {self.num_generations}")
            print(f"  - Num Eval Episodes/Chromosome (Optimization): {self.num_optimize_eval_episodes}")
            
            start_time = time.time()
            self.best_chromosome = self._optimize_policy_ga()
            end_time = time.time()
            print(f"\nGA Optimization finished in {end_time - start_time:.2f} seconds.")
            print(f"Optimized GA Meta-Heuristic Policy (Best Chromosome) Found:\n{self.best_chromosome}")

            if self.save_policy_path:
                print(f"\n--- Saving Optimized GA Meta-Heuristic Policy ---")
                try:
                    os.makedirs(os.path.dirname(self.save_policy_path), exist_ok=True)
                    with open(self.save_policy_path, 'w') as f:
                        json.dump(self.best_chromosome, f, indent=4)
                    print(f"Optimized GA policy saved to: {self.save_policy_path}")
                except Exception as e:
                    print(f"Error saving GA policy to '{self.save_policy_path}': {e}", file=sys.stderr)

    def _generate_random_gene(self, item_idx):
        valid_suppliers = np.where(self.item_supplier_matrix[item_idx, :] == 1)[0]
        if not valid_suppliers.size: # Should not happen in a well-defined environment
            print(f"Warning: Item {item_idx} has no valid suppliers. Defaulting to supplier 0 and COP with 0 quantity.")
            return (0, HEURISTIC_COP, 0.0) 
            
        supplier_idx = self.ga_rng.choice(valid_suppliers)
        heuristic_id = self.ga_rng.choice([HEURISTIC_COP, HEURISTIC_BSP, HEURISTIC_BSPEW])
        
        param_value = 0.0
        if heuristic_id == HEURISTIC_COP:
            if self.quantity_options:
                param_value = self.ga_rng.choice(self.quantity_options)
            else: # Fallback if options not provided
                param_value = self.ga_rng.integers(0, 10) 
        elif heuristic_id == HEURISTIC_BSP or heuristic_id == HEURISTIC_BSPEW:
            if self.base_stock_level_options:
                param_value = self.ga_rng.choice(self.base_stock_level_options)
            else: # Fallback
                param_value = self.ga_rng.integers(0, 50)
        
        return (int(supplier_idx), int(heuristic_id), float(param_value))

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = [self._generate_random_gene(i) for i in range(self.n_items)]
            population.append(chromosome)
        return population

    def _calculate_fitness(self, chromosome, seed_batch_key: int | None = None):
        total_reward_across_episodes = 0.0
        if self.num_optimize_eval_episodes == 0: return -np.inf

        # CRN: generate shared seeds per fitness evaluation call
        base = getattr(self.env, '_initial_seed', None)
        base = int(base) if base is not None else 0
        mix_key = int(seed_batch_key) if seed_batch_key is not None else 0
        mixed = (base ^ ((mix_key * 0x9E3779B1) & 0x7FFFFFFF) ^ 0x00A5A5A5) & 0x7FFFFFFF
        rng_local = np.random.default_rng(mixed)
        eval_seeds = [int(s) for s in rng_local.integers(low=0, high=2**31 - 1, size=self.num_optimize_eval_episodes)]

        for seed in eval_seeds:
            observation, _ = self.env.reset(seed=int(seed)) # Env is reset for each episode
            terminated = False
            truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action = self._get_action_from_chromosome(chromosome) # Pass current chromosome
                observation_next, reward, terminated, truncated, info = self.env.step(action, verbose=False)
                episode_reward += reward
                observation = observation_next # Not strictly needed if action doesn't depend on obs, but good practice
            total_reward_across_episodes += episode_reward
        return total_reward_across_episodes / self.num_optimize_eval_episodes

    def _selection(self, population, fitnesses):
        # Tournament selection
        selected_parents = []
        for _ in range(self.population_size): # Select population_size parents
            tournament_indices = self.ga_rng.choice(len(population), size=self.tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx_in_tournament = np.argmax(tournament_fitnesses)
            selected_parents.append(population[tournament_indices[winner_idx_in_tournament]])
        return selected_parents

    def _crossover(self, parent1, parent2):
        if self.ga_rng.random() > self.crossover_rate:
            return parent1[:], parent2[:] # Return copies
        
        # Single point crossover
        if self.n_items <= 1: # Cannot do crossover if only one item
            return parent1[:], parent2[:]

        # Chromosome is a list of genes (tuples)
        # Choose a crossover point between 1 and n_items - 1
        point = self.ga_rng.integers(1, self.n_items) 
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutation(self, chromosome):
        mutated_chromosome = list(chromosome) # Make a mutable copy
        for i in range(self.n_items):
            if self.ga_rng.random() < self.mutation_rate:
                # Mutate the gene for item i by generating a new random one
                mutated_chromosome[i] = self._generate_random_gene(i)
        return mutated_chromosome

    def _optimize_policy_ga(self):
        population = self._initialize_population()
        best_chromosome_overall = None
        best_fitness_overall = -np.inf

        if 'tqdm' in sys.modules:
            generation_iterator = tqdm(range(self.num_generations), desc="GA Optimizing", unit="gen")
        else:
            generation_iterator = range(self.num_generations)
            print("GA Optimizing...")

        for gen in generation_iterator:
            # Vary seeds across generations by offsetting env._initial_seed deterministically
            original_initial_seed = getattr(self.env, '_initial_seed', None)
            try:
                base = int(original_initial_seed) if original_initial_seed is not None else 0
            except Exception:
                base = 0
            # Mix gen into base to vary seeds across generations
            mixed = (base ^ (gen * 0x9E3779B1)) & 0x7FFFFFFF
            setattr(self.env, '_initial_seed', mixed)

            fitnesses = [self._calculate_fitness(chromo, seed_batch_key=i) for i, chromo in enumerate(population)]

            # Restore original seed after fitness evaluation for this generation
            setattr(self.env, '_initial_seed', original_initial_seed)

            current_best_fitness_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_fitness_idx]

            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                best_chromosome_overall = population[current_best_fitness_idx][:] # Store a copy
                if 'tqdm' in sys.modules and hasattr(generation_iterator, 'set_postfix'):
                    generation_iterator.set_postfix({"Best Fitness": f"{best_fitness_overall:.2f}"})
            
            # Elitism: Keep the best chromosome from current generation if it's good
            new_population = []
            if best_chromosome_overall: # Make sure one best is found
                 new_population.append(best_chromosome_overall[:]) 

            # Selection
            parents = self._selection(population, fitnesses)
            
            # Crossover and Mutation
            # Ensure new_population reaches population_size
            # Current simple elitism: just add the best overall. A better elitism might preserve k best.
            # Here, parents are selected from old population, crossover and mutation generate new children.
            # The new population will be formed by these children.
            
            children = []
            for i in range(0, len(parents) -1 , 2): # Iterate over pairs of parents
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                children.append(self._mutation(child1))
                children.append(self._mutation(child2))
            
            # Fill up the population, potentially with some parents if not enough children, or truncate
            if len(new_population) < self.population_size:
                children_needed = self.population_size - len(new_population)
                new_population.extend(children[:children_needed])

            population = new_population
            # Ensure population size is maintained (can happen if odd number of parents/children or elitism choice)
            while len(population) < self.population_size:
                population.append(self._generate_random_gene(self.ga_rng.integers(0,self.n_items)) if self.n_items > 0 else []) # Add random individual if needed
            population = population[:self.population_size] # Truncate if too many

        if best_chromosome_overall is None and population: # If no improvement, pick best from last pop
            fitnesses = [self._calculate_fitness(chromo) for chromo in population]
            best_chromosome_overall = population[np.argmax(fitnesses)][:]
            best_fitness_overall = np.max(fitnesses)
            print(f"GA Warning: No improvement over initial random. Final best fitness: {best_fitness_overall:.2f}")
        elif not population and not best_chromosome_overall:
            print("GA Error: Population is empty and no best chromosome found. Returning a default.")
            # Create a default chromosome (all items COP, supplier 0, qty 0)
            default_gene = (0, HEURISTIC_COP, 0.0)
            best_chromosome_overall = [default_gene for _ in range(self.n_items)]

        print(f"\nGA Optimization complete. Best Fitness found (opt): {best_fitness_overall:.2f}")
        return best_chromosome_overall

    def _get_action_from_chromosome(self, chromosome):
        action_matrix = np.zeros((self.n_items, self.n_suppliers), dtype=np.float32)
        # We need outstanding orders for BSP and BSPEW, calculate once
        current_outstanding_orders_all_items = self._calculate_outstanding_orders()

        for i in range(self.n_items):
            if i >= len(chromosome): # Should not happen with correct chromosome
                print(f"Warning: Chromosome shorter than n_items. Skipping item {i}.")
                continue

            supplier_idx, heuristic_id, param_value = chromosome[i]
            
            order_qty = 0.0
            if heuristic_id == HEURISTIC_COP:
                order_qty = param_value # param_value is the constant order quantity
            
            elif heuristic_id == HEURISTIC_BSP or heuristic_id == HEURISTIC_BSPEW:
                current_inv_item = self.env.inventory_level[i]
                outstanding_item = current_outstanding_orders_all_items[i]
                inv_pos = current_inv_item + outstanding_item
                
                if heuristic_id == HEURISTIC_BSP:
                    order_qty = param_value - inv_pos
                else: # HEURISTIC_BSPEW
                    ew_item_i = 0.0
                    chosen_supplier_lead_time = int(self.env.lead_times[i, supplier_idx]) # Correct supplier_idx for item i
                    
                    if self.bspew_waste_estimation_method == "closed_form_approx":
                        ew_item_i = self._calculate_ew_closed_form_approx(i, chosen_supplier_lead_time)
                    elif self.bspew_waste_estimation_method == "deterministic_simulation":
                        ew_item_i = self._calculate_ew_deterministic_simulation(i, chosen_supplier_lead_time)
                    else:
                        print(f"Warning: Unknown BSPEW method '{self.bspew_waste_estimation_method}'. Using 0 EW for item {i}.")
                    
                    order_qty = param_value - inv_pos + ew_item_i
            
            action_matrix[i, supplier_idx] = float(max(0.0, order_qty))
            
        return action_matrix

    # --- Helper methods for BSP/BSPEW calculations (adapted from BaseStockPolicyAgent & BSPEWAgent) ---
    def _calculate_outstanding_orders(self) -> np.ndarray: # From BaseStockPolicyAgent
        outstanding = np.zeros(self.env.n_items, dtype=int)
        current_time = self.env.current_step
        lead_times = self.env.lead_times
        # Assuming self.env.order_history is available and structured as in BaseStockPolicyAgent
        if hasattr(self.env, 'order_history') and self.env.order_history is not None:
            for t_placed, i_item, s_supplier, qty_ordered in self.env.order_history:
                if i_item < self.n_items and s_supplier < self.n_suppliers : # Basic check
                    arrival_time = t_placed + lead_times[i_item, s_supplier]
                    if arrival_time > current_time:
                        outstanding[i_item] += qty_ordered
        return outstanding

    # --- EW Calculation methods (adapted from BSPEWAgent) ---
    # These methods need access to self.env (for inventory_age, max_age, shelf_life_cdf, stoch_model, lead_times)
    # and self.ew_sim_rng, self.bspew_num_ew_demand_sim_paths, etc.

    def _estimate_future_daily_demands_mc(self, item_idx: int, horizon: int) -> np.ndarray:
        # Adapted from BSPEWAgent
        if horizon <= 0 or self.bspew_num_ew_demand_sim_paths <= 0:
            return np.zeros(horizon)

        all_simulated_demands_for_item = np.zeros((self.bspew_num_ew_demand_sim_paths, horizon))
        
        # Temporarily use ew_sim_rng for stoch_model if it has an rng attribute
        original_stoch_model_rng = None
        stoch_model_had_rng_attr = hasattr(self.env.stoch_model, 'rng')
        if stoch_model_had_rng_attr:
            original_stoch_model_rng = self.env.stoch_model.rng
            self.env.stoch_model.rng = self.ew_sim_rng # Use GA agent's ew_sim_rng

        for i_path in range(self.bspew_num_ew_demand_sim_paths):
            try:
                # generate_scenario might take n_time_steps or (n_items, n_time_steps)
                # We need demands for a specific item over the horizon
                full_scenario = self.env.stoch_model.generate_scenario(n_time_steps=horizon)
                
                if full_scenario.ndim == 2 and full_scenario.shape[0] == self.env.n_items and full_scenario.shape[1] == horizon:
                    all_simulated_demands_for_item[i_path, :] = full_scenario[item_idx, :]
                elif full_scenario.ndim == 1 and self.env.n_items == 1 and full_scenario.shape[0] == horizon: # Single item case
                    all_simulated_demands_for_item[i_path, :] = full_scenario[:]
                elif full_scenario.ndim == 1 and full_scenario.shape[0] == self.env.n_items * horizon : # Another possible flat format
                    all_simulated_demands_for_item[i_path, :] = full_scenario.reshape(self.env.n_items, horizon)[item_idx,:]
                else: # Fallback if shape is unexpected.
                    # Try to generate for single item if stoch_model supports it
                    if hasattr(self.env.stoch_model, 'generate_demands_for_item'):
                         all_simulated_demands_for_item[i_path, :] = self.env.stoch_model.generate_demands_for_item(item_idx, horizon)
                    else:
                        print(f"Warning (GA EW Demand MC): stoch_model.generate_scenario returned unexpected shape "
                              f"{full_scenario.shape}. Using zeros for item {item_idx}, path {i_path}.")
                        all_simulated_demands_for_item[i_path, :] = np.zeros(horizon)
            except Exception as e:
                print(f"Error (GA EW Demand MC): during stoch_model.generate_scenario for item {item_idx}: {e}. Using zeros for path {i_path}.")
                all_simulated_demands_for_item[i_path, :] = np.zeros(horizon)
        
        if stoch_model_had_rng_attr and original_stoch_model_rng is not None:
            self.env.stoch_model.rng = original_stoch_model_rng # Restore original RNG

        avg_daily_demands = np.mean(all_simulated_demands_for_item, axis=0)
        return avg_daily_demands

    def _calculate_ew_deterministic_simulation(self, item_idx: int, chosen_supplier_lead_time: int) -> float:
        # Adapted from BSPEWAgent
        current_inventory_sim = self.env.inventory_age[item_idx, :].astype(float)
        max_age_M = self.env.max_age

        sim_horizon = self.bspew_waste_horizon_review_periods + chosen_supplier_lead_time - 1
        if sim_horizon <= 0:
            return 0.0

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
                if prob_survival_up_to_age > 1e-9: # Avoid division by zero
                    prob_expire_in_next_step_sim = np.clip((cdf_age_plus_1 - cdf_age) / prob_survival_up_to_age, 0.0, 1.0)
                elif cdf_age_plus_1 > cdf_age : # If it cannot survive up to current age, it will expire if it has a chance to age further
                    prob_expire_in_next_step_sim = 1.0


                wasted_count_sim_float = n_items_in_bin_sim_float * prob_expire_in_next_step_sim
                simulated_waste_this_step_float += wasted_count_sim_float
                survivors_count_sim_float = n_items_in_bin_sim_float - wasted_count_sim_float

                if age_idx_sim < max_age_M - 1 and survivors_count_sim_float > 1e-9:
                    next_step_inventory_sim[age_idx_sim + 1] += survivors_count_sim_float
            
            current_inventory_sim = next_step_inventory_sim
            total_simulated_waste_float += simulated_waste_this_step_float

            demand_to_satisfy_sim_float = demand_for_this_sim_day_float
            for age_idx_sim in range(max_age_M - 1, -1, -1): # FIFO
                if demand_to_satisfy_sim_float <= 1e-9: break
                
                available_in_bin_sim_float = current_inventory_sim[age_idx_sim]
                if available_in_bin_sim_float <= 1e-9: continue
                
                fulfilled_from_bin_sim_float = min(available_in_bin_sim_float, demand_to_satisfy_sim_float)
                current_inventory_sim[age_idx_sim] -= fulfilled_from_bin_sim_float
                demand_to_satisfy_sim_float -= fulfilled_from_bin_sim_float
        
        return float(total_simulated_waste_float)

    def _calculate_ew_closed_form_approx(self, item_idx: int, chosen_supplier_lead_time: int) -> float:
        # Adapted from BSPEWAgent
        current_inventory_by_age = self.env.inventory_age[item_idx, :].astype(float)
        max_shelf_life_M = self.env.max_age
        sim_horizon = self.bspew_waste_horizon_review_periods + chosen_supplier_lead_time - 1
        if sim_horizon <= 0: return 0.0

        future_daily_demands_est = self._estimate_future_daily_demands_mc(item_idx, sim_horizon)
        if future_daily_demands_est.size == 0 :
            mean_demand_mu_for_horizon = 0.0
        else:
            mean_demand_mu_for_horizon = np.mean(future_daily_demands_est) if len(future_daily_demands_est) > 0 else 0.0


        sum_old_stock_float = 0.0
        start_age_for_old_stock = max(0, int(np.ceil(max_shelf_life_M - sim_horizon)))
        
        for age_k in range(start_age_for_old_stock, int(max_shelf_life_M)):
            if age_k < current_inventory_by_age.shape[0]:
                 sum_old_stock_float += current_inventory_by_age[age_k]

        total_demand_for_horizon_est = mean_demand_mu_for_horizon * sim_horizon
        ew_fifo = max(0.0, sum_old_stock_float - total_demand_for_horizon_est)
        return float(ew_fifo)

    def run(self, render_steps=False, verbose=False):
        all_episode_rewards = []
        if self.best_chromosome is None:
             print("Error: GA Meta-Heuristic Agent has no optimized policy to run. Exiting.", file=sys.stderr)
             return []

        print(f"\nRunning final evaluation with {'Loaded' if self.load_policy_path else 'Optimized'} GA Meta-Heuristic Policy "
              f"for {self.num_final_eval_episodes} episode(s)...")
        
        for episode_idx in range(self.num_final_eval_episodes):
            if self.logger:
                self.logger.start_episode(episode_num=episode_idx)

            observation, info_reset = self.env.reset()
            terminated = False
            truncated = False
            total_reward_episode = 0.0
            
            while not (terminated or truncated):
                current_step_env = self.env.current_step
                action_to_take = self._get_action_from_chromosome(self.best_chromosome)
                
                observation_next, reward_step, terminated, truncated, info_step = self.env.step(action_to_take, verbose=verbose)
                
                if self.logger:
                    should_log_action = self.logger.log_actions if self.logger else False
                    self.logger.log_step(
                        step_num=current_step_env,
                        reward=reward_step,
                        info=info_step,
                        action=action_to_take if should_log_action else None
                    )

                if render_steps: self.env.render()
                total_reward_episode += reward_step
                observation = observation_next
            
            if self.logger:
                self.logger.end_episode()
            
            if verbose: print(f"Evaluation Episode {episode_idx + 1}: Total Reward: {total_reward_episode:.2f}")
            all_episode_rewards.append(total_reward_episode)

        if self.logger:
            self.logger.finalize_logs()

        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(f"Average reward over {self.num_final_eval_episodes} final evaluation episodes (GA Meta-Heuristic): {avg_final_reward:.2f}")
        return all_episode_rewards