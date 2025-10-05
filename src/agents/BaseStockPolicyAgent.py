# src/agents/BaseStockPolicyAgent.py
import numpy as np
from typing import Optional
import time
import sys
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

try:
    from src.utils.simulation_logger import SimulationLogger
except ImportError:
    print("Warning: SimulationLogger utility not found. Detailed step logging will be disabled.")
    SimulationLogger = None

class BaseStockPolicyAgent:
    def __init__(self, env,
                 num_candidate_policies: int = 100,
                 num_optimize_eval_episodes: int = 10,
                 num_final_eval_episodes: int = 1,
                 base_stock_level_options: list = [5, 10, 15, 20, 25, 30, 40, 50],
                 load_policy_path: str = None,
                 save_policy_path: str = None,
                 logger_settings: dict = None,
                 optimize_on_init: bool = True,
                 **other_kwargs):

        self.env = env
        self.optimized_policy = None
        self.num_candidate_policies = num_candidate_policies
        self.num_optimize_eval_episodes = num_optimize_eval_episodes
        self.num_final_eval_episodes = num_final_eval_episodes
        self.base_stock_level_options = base_stock_level_options
        self.load_policy_path = load_policy_path
        self.save_policy_path = save_policy_path
        self.optimized_policy = None

        self.logger = None
        if SimulationLogger and logger_settings and logger_settings.get("log_step_details", False):
            exp_name_for_logger = logger_settings.get("experiment_name", "default_experiment_unknown_seed")
            self.logger = SimulationLogger(
                log_dir=logger_settings.get("log_dir", "./src/results/simulation_logs"),
                experiment_name=exp_name_for_logger,
                log_step_details=logger_settings.get("log_step_details", True),
                log_actions=logger_settings.get("log_actions", False),
                n_items=self.env.n_items,
                n_suppliers=self.env.n_suppliers
            )
            if self.logger.log_step_details_enabled:
                 print(f"Detailed simulation logging enabled. Log file: {self.logger.log_file_path}")

        if self.load_policy_path:
            print(f"\n--- Loading Pre-optimized BSP Policy ---")
            try:
                self.optimized_policy = np.load(self.load_policy_path)
                print(f"Policy successfully loaded from: {self.load_policy_path}")
                print(f"Loaded Policy (Base Stock Levels per Item/Supplier):\n{self.optimized_policy}")
                if self.optimized_policy.shape != self.env.action_space.shape:
                     raise ValueError(f"Loaded policy shape {self.optimized_policy.shape} incompatible with env action space {self.env.action_space.shape}")
                for i in range(self.env.n_items):
                    if np.count_nonzero(self.optimized_policy[i, :]) > 1:
                        print(f"Warning: Loaded policy for item {i} has multiple suppliers defined. Using first found.")
            except Exception as e:
                print(f"Error loading policy from '{self.load_policy_path}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Optimizing Base Stock Policy (Single Supplier per Item Variant) using Monte Carlo...")
            print(f"  - Num Candidate Policies: {self.num_candidate_policies}")
            print(f"  - Eval Episodes/Policy (Optimization): {self.num_optimize_eval_episodes}")
            print(f"  - Base Stock Level Options: {self.base_stock_level_options}")

            start_time = time.time()
            self.optimized_policy = self._optimize_bsp()
            end_time = time.time()
            print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")
            print(f"Optimized Base Stock Policy Found (Levels per Item/Supplier):\n{self.optimized_policy}")

            if self.save_policy_path:
                print(f"\n--- Saving Optimized BSP Policy ---")
                try:
                    os.makedirs(os.path.dirname(self.save_policy_path), exist_ok=True)
                    np.save(self.save_policy_path, self.optimized_policy)
                    print(f"Optimized policy saved to: {self.save_policy_path}")
                except Exception as e:
                    print(f"Error saving policy to '{self.save_policy_path}': {e}", file=sys.stderr)

    def _calculate_outstanding_orders(self) -> np.ndarray:
        outstanding = np.zeros(self.env.n_items, dtype=int) # outstanding quantity is discrete
        current_time = self.env.current_step
        lead_times = self.env.lead_times
        for t_placed, i, s, qty_ordered in self.env.order_history:
             arrival_time = t_placed + lead_times[i, s]
             if arrival_time > current_time: # Order arrives in a future step
                   outstanding[i] += qty_ordered
        return outstanding

    def _get_action_from_policy(self, bsp_policy_matrix: np.ndarray) -> np.ndarray:
        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        current_inventory_level = self.env.inventory_level # Array of on-hand inventory levels
        outstanding_orders = self._calculate_outstanding_orders()

        for i in range(self.env.n_items):
            chosen_supplier_idx = np.where(bsp_policy_matrix[i, :] > 0)[0]
            if len(chosen_supplier_idx) > 0:
                s = chosen_supplier_idx[0]
                # Ensure all components are float for the calculation
                target_level_float = float(bsp_policy_matrix[i, s])
                on_hand_float = float(current_inventory_level[i])
                outstanding_float = float(outstanding_orders[i])

                # Standard base stock policy calculation
                inventory_position = on_hand_float + outstanding_float
                order_qty = target_level_float - inventory_position
                
                action[i, s] = float(max(0.0, order_qty)) # Order non-negative quantity
        return action

    def _evaluate_policy(self, policy_matrix: np.ndarray, seed_batch_key: Optional[int] = None) -> float:
        total_reward_across_episodes = 0.0
        if self.num_optimize_eval_episodes == 0: return -np.inf # Avoid division by zero

        # Common Random Numbers: generate a fixed list of seeds for this evaluation call
        base = getattr(self.env, '_initial_seed', None)
        base = int(base) if base is not None else 0
        # Vary per-call seed set using seed_batch_key (e.g., candidate idx or generation)
        mix_key = int(seed_batch_key) if seed_batch_key is not None else 0
        mixed = (base ^ ((mix_key * 0x9E3779B1) & 0x7FFFFFFF) ^ 0x00BADC0DE) & 0x7FFFFFFF
        rng_local = np.random.default_rng(mixed)
        eval_seeds = [int(s) for s in rng_local.integers(low=0, high=2**31 - 1, size=self.num_optimize_eval_episodes)]

        for seed in eval_seeds:
            observation, _ = self.env.reset(seed=int(seed))
            terminated = False
            truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action = self._get_action_from_policy(policy_matrix)
                observation_next, reward, terminated, truncated, info = self.env.step(action, verbose=False)
                episode_reward += reward
                observation = observation_next
            total_reward_across_episodes += episode_reward
        return total_reward_across_episodes / self.num_optimize_eval_episodes

    def _generate_random_bsp(self) -> np.ndarray:
        n_items = self.env.n_items
        n_suppliers = self.env.n_suppliers
        item_supplier_matrix = self.env.item_supplier_matrix
        candidate_bsp = np.zeros((n_items, n_suppliers), dtype=np.float32)

        for i in range(n_items):
            valid_suppliers_for_i = np.where(item_supplier_matrix[i, :] == 1)[0]
            if not valid_suppliers_for_i.size: continue # No valid suppliers for this item

            s_chosen = self.env.env_rng.choice(valid_suppliers_for_i)
            
            if not self.base_stock_level_options: 
                chosen_level = 0.0 # Default if no options
            else: 
                chosen_level = self.env.env_rng.choice(self.base_stock_level_options)
            candidate_bsp[i, s_chosen] = float(chosen_level)
        return candidate_bsp

    def _optimize_bsp(self) -> np.ndarray:
        best_avg_reward = -np.inf
        # Initialize with a zero policy or a very basic one
        best_bsp_policy = np.zeros_like(self.env.action_space.sample(), dtype=np.float32) 
        if not self.base_stock_level_options and self.num_candidate_policies > 0:
            print("Warning: base_stock_level_options is empty. Optimization will likely result in a zero policy.")

        if self.num_candidate_policies == 0 :
            print("Warning: num_candidate_policies is 0. Returning initial zero policy.")
            return best_bsp_policy


        progress_bar_desc = f"Optimizing BSP ({self.env.spec.id if self.env.spec else 'env'})"
        if 'tqdm' in sys.modules:
             iterator = tqdm(range(self.num_candidate_policies), desc=progress_bar_desc, unit="policy")
        else: 
            iterator = range(self.num_candidate_policies)
            print(progress_bar_desc + "...")


        for iter_idx, _ in enumerate(iterator):
            candidate_bsp = self._generate_random_bsp()
            avg_reward = self._evaluate_policy(candidate_bsp, seed_batch_key=iter_idx)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_bsp_policy = candidate_bsp.copy()
                if 'tqdm' in sys.modules and hasattr(iterator, 'set_postfix'):
                     iterator.set_postfix({"Best Avg Reward": f"{best_avg_reward:.2f}"}, refresh=True)
        
        if best_avg_reward == -np.inf: 
            print("\nWarning: No valid policy improved initial reward (-inf). This might happen if num_optimize_eval_episodes is 0 or options are very limited.")
        else: 
            print(f"\nOptimization complete. Best Avg Reward (opt): {best_avg_reward:.2f}")
        return best_bsp_policy

    def run(self, render_steps=False, verbose=False):
        all_episode_rewards = []
        if self.optimized_policy is None:
             print("Error: Agent has no optimized policy to run. Consider increasing num_candidate_policies or checking base_stock_level_options. Exiting.", file=sys.stderr)
             return []

        print(f"\nRunning final evaluation with {'Loaded' if self.load_policy_path else 'Optimized'} BSP for {self.num_final_eval_episodes} episode(s)...")
        for episode_idx in range(self.num_final_eval_episodes):
            if self.logger:
                self.logger.start_episode(episode_num=episode_idx)

            observation, info_reset = self.env.reset()
            terminated = False
            truncated = False
            total_reward_episode = 0.0

            while not (terminated or truncated):
                current_step_env = self.env.current_step 
                action = self._get_action_from_policy(self.optimized_policy)
                observation_next, reward_step, terminated, truncated, info_step = self.env.step(action, verbose=verbose)
                
                if self.logger:
                    should_log_action = self.logger.log_actions if self.logger else False
                    self.logger.log_step(
                        step_num=current_step_env,
                        reward=reward_step, 
                        info=info_step, 
                        action=action if should_log_action else None
                    )

                if render_steps: self.env.render()
                total_reward_episode += reward_step
                observation = observation_next
            
            if self.logger: self.logger.end_episode()
            if verbose: print(f"Evaluation Episode {episode_idx + 1}: Total Reward: {total_reward_episode:.2f}")
            all_episode_rewards.append(total_reward_episode)

        if self.logger: self.logger.finalize_logs()

        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(f"Average reward over {self.num_final_eval_episodes} final evaluation episodes: {avg_final_reward:.2f}")
        else:
            print("No final evaluation episodes run.")
        return all_episode_rewards