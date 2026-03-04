from src.agents.BSPEWAgent import BSPEWAgent # Assuming BSPEWAgent is in this path
import numpy as np
import sys
import os
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable


class BSPEWLowAgent(BSPEWAgent):
    def __init__(self, env,
                 s1_options: list = None,
                 s2_options: list = None,
                 b_options: list = None,
                 # These are from BaseStockPolicyAgent and BSPEWAgent
                 num_candidate_policies: int = 100, # Default from BaseStockPolicyAgent
                 num_optimize_eval_episodes: int = 10, # Default from BaseStockPolicyAgent
                 num_final_eval_episodes: int = 1, # Default from BaseStockPolicyAgent
                 load_policy_path: str = None, # Will be handled by BSPEWLowAgent specifically
                 save_policy_path: str = None, # Will be handled by BSPEWLowAgent specifically
                 logger_settings: dict = None, # From BaseStockPolicyAgent
                 waste_estimation_method: str = "deterministic_simulation", # From BSPEWAgent
                 waste_horizon_review_periods: int = 1, # From BSPEWAgent
                 num_ew_demand_sim_paths: int = 30, # From BSPEWAgent
                 **other_kwargs): # Catch any other kwargs for super classes

        self.s1_options = s1_options if s1_options is not None else [0, 5, 10, 15, 20, 25, 30]
        self.s2_options = s2_options if s2_options is not None else [5, 10, 15, 20, 25, 30, 40, 50]
        self.b_options = b_options if b_options is not None else [1, 5, 10, 15, 20, 25, 30]

        
        # We construct `super_kwargs` explicitly to control what BaseStockPolicyAgent sees
        super_kwargs = {
            'num_candidate_policies': num_candidate_policies,
            'num_optimize_eval_episodes': num_optimize_eval_episodes,
            'num_final_eval_episodes': num_final_eval_episodes,
            'logger_settings': logger_settings,
            'waste_estimation_method': waste_estimation_method,
            'waste_horizon_review_periods': waste_horizon_review_periods,
            'num_ew_demand_sim_paths': num_ew_demand_sim_paths,
            'load_policy_path': None, # Explicitly None for super call
            'save_policy_path': None, # Explicitly None for super call
            'optimize_on_init': False,
            'base_stock_level_options': [], # Ensure it doesn't use this
            **other_kwargs
        }

        # Call BSPEWAgent's __init__ which calls BaseStockPolicyAgent's __init__
        super().__init__(env, **super_kwargs)

        self.load_policy_path = load_policy_path # Use the original passed value
        self.save_policy_path = save_policy_path # Use the original passed value
        
        # Also ensure these are set on self for _optimize_bsp_low_ew if not passed via kwargs
        self.num_candidate_policies = num_candidate_policies
        self.num_optimize_eval_episodes = num_optimize_eval_episodes
        self.num_final_eval_episodes = num_final_eval_episodes


        if self.load_policy_path:
            print(f"\n--- Loading Pre-optimized BSP-low-EW Policy ---")
            try:
                loaded_data = np.load(self.load_policy_path)
                # Expected shape (n_items, n_suppliers, 4)
                if loaded_data.ndim != 3 or \
                   loaded_data.shape[0] != self.env.n_items or \
                   loaded_data.shape[1] != self.env.n_suppliers or \
                   loaded_data.shape[2] != 4:
                    raise ValueError(f"Loaded policy shape {loaded_data.shape} "
                                     f"incompatible with expected BSP-low-EW shape "
                                     f"({self.env.n_items}, {self.env.n_suppliers}, 4)")
                self.optimized_policy = loaded_data # THIS IS THE 4D policy
                print(f"BSP-low-EW Policy successfully loaded from: {self.load_policy_path}")
                # ... (rest of your loading printout logic)
            except Exception as e:
                print(f"Error loading BSP-low-EW policy from '{self.load_policy_path}': {e}", file=sys.stderr)
                # Decide if to exit or proceed to optimize
                print("Proceeding to optimize BSP-low-EW policy instead.")
                self._reset_env_seed_sequence()
                self.optimized_policy = self._optimize_bsp_low_ew() # Fallback to optimize
        else:
            # If not loading, then optimize specifically for BSPELowAgent
            print("Optimizing BSP-low-EW Policy (Single Supplier per Item Variant) using Monte Carlo...")
            # num_candidate_policies etc. should be available as self attributes now
            print(f"  - Num Candidate Policies: {self.num_candidate_policies}")
            print(f"  - Eval Episodes/Policy (Optimization): {self.num_optimize_eval_episodes}")
            print(f"  - S1 options: {self.s1_options}")
            print(f"  - S2 options: {self.s2_options}")
            print(f"  - b options: {self.b_options}")

            start_time = time.time()
            self._reset_env_seed_sequence()
            self.optimized_policy = self._optimize_bsp_low_ew()
            end_time = time.time()
            # ... (rest of your optimization printout and save logic) ...
            print(f"\nBSP-low-EW Optimization finished in {end_time - start_time:.2f} seconds.")
            if self.optimized_policy is not None and np.sum(self.optimized_policy[:,:,3]) > 0:
                 print("Optimized BSP-low-EW Policy Found:")
                 for i in range(self.env.n_items):
                    active_suppliers = np.where(self.optimized_policy[i, :, 3] == 1)[0]
                    if len(active_suppliers) == 1:
                        s = active_suppliers[0]
                        print(f"  Item {i}, Supplier {s}: S1={self.optimized_policy[i,s,0]:.2f}, S2={self.optimized_policy[i,s,1]:.2f}, b={self.optimized_policy[i,s,2]:.2f}")
            else:
                print("Warning: BSP-low-EW optimization did not result in a valid policy.")


            if self.save_policy_path and self.optimized_policy is not None:
                print(f"\n--- Saving Optimized BSP-low-EW Policy ---")
                try:
                    save_dir = os.path.dirname(self.save_policy_path)
                    if save_dir: # Ensure directory exists only if save_dir is not empty (e.g. saving in current dir)
                        os.makedirs(save_dir, exist_ok=True)
                    np.save(self.save_policy_path, self.optimized_policy)
                    print(f"Optimized BSP-low-EW policy saved to: {self.save_policy_path}")
                except Exception as e:
                    print(f"Error saving BSP-low-EW policy to '{self.save_policy_path}': {e}", file=sys.stderr)


    def _generate_random_bsp_low_ew_policy(self) -> np.ndarray:
        n_items = self.env.n_items
        n_suppliers = self.env.n_suppliers
        item_supplier_matrix = self.env.item_supplier_matrix
        # policy_params: [S1, S2, b, active_flag]
        candidate_policy = np.zeros((n_items, n_suppliers, 4), dtype=np.float32)

        for i in range(n_items):
            valid_suppliers_for_i = np.where(item_supplier_matrix[i, :] == 1)[0]
            if not valid_suppliers_for_i.size:
                continue

            # Allow multiple suppliers per item
            n_chosen = int(self.env.env_rng.integers(1, len(valid_suppliers_for_i) + 1))
            chosen_suppliers = self.env.env_rng.choice(valid_suppliers_for_i, size=n_chosen, replace=False)

            for s_chosen in chosen_suppliers:
                candidate_policy[i, s_chosen, 3] = 1.0 # Mark as active supplier

                chosen_s1 = 0.0
                if self.s1_options:
                    chosen_s1 = float(self.env.env_rng.choice(self.s1_options))
                candidate_policy[i, s_chosen, 0] = chosen_s1

                chosen_s2 = 0.0
                if self.s2_options:
                    chosen_s2 = float(self.env.env_rng.choice(self.s2_options))
                candidate_policy[i, s_chosen, 1] = chosen_s2

                chosen_b = 1.0 # b must be > 0 for alpha calculation
                if self.b_options:
                    valid_b_options = [b_val for b_val in self.b_options if b_val > 1e-6] # Ensure b is positive
                    if valid_b_options:
                        chosen_b = float(self.env.env_rng.choice(valid_b_options))
                    else:
                        print("Warning: No valid b_options > 0 found. Defaulting b to 1.0.")
                candidate_policy[i, s_chosen, 2] = chosen_b

        return candidate_policy
    
    def _optimize_bsp(self) -> np.ndarray:
        print("BSPELowAgent's _optimize_bsp override: Returning dummy policy. Actual optimization follows.")
        return np.zeros((self.env.n_items, self.env.n_suppliers, 4), dtype=np.float32)

    def _optimize_bsp_low_ew(self) -> np.ndarray:
        best_avg_reward = -np.inf
        # policy_params: [S1, S2, b, active_flag]
        best_policy = np.zeros((self.env.n_items, self.env.n_suppliers, 4), dtype=np.float32)

        if self.num_candidate_policies == 0:
            print("Warning: num_candidate_policies is 0. Returning initial zero BSP-low-EW policy.")
            return best_policy
        if not self.s1_options or not self.s2_options or not self.b_options:
            print("Warning: S1, S2, or b options are empty. Optimization will likely result in a default policy.")


        progress_bar_desc = f"Optimizing BSP-low-EW ({self.env.spec.id if self.env.spec else 'env'})"
        if 'tqdm' in sys.modules:
            iterator = tqdm(range(self.num_candidate_policies), desc=progress_bar_desc, unit="policy")
        else:
            iterator = range(self.num_candidate_policies)
            print(progress_bar_desc + "...")

        for iter_idx, _ in enumerate(iterator):
            candidate_policy = self._generate_random_bsp_low_ew_policy()
            # Uses BaseStockPolicyAgent._evaluate_policy with CRN; pass iter_idx to vary seed set per candidate
            avg_reward = self._evaluate_policy(candidate_policy, seed_batch_key=iter_idx)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_policy = candidate_policy.copy()
                if 'tqdm' in sys.modules and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({"Best Avg Reward": f"{best_avg_reward:.2f}"}, refresh=True)

        if best_avg_reward == -np.inf:
            print("\nWarning: No valid BSP-low-EW policy improved initial reward (-inf).")
        else:
            print(f"\nBSP-low-EW Optimization complete. Best Avg Reward (opt): {best_avg_reward:.2f}")
        return best_policy

    def _get_action_from_policy(self, bsp_low_ew_policy_matrix: np.ndarray) -> np.ndarray:
        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        current_inventory_level_total = self.env.inventory_level # Total on-hand for each item
        outstanding_orders_total = self._calculate_outstanding_orders() # Total outstanding for each item

        for i in range(self.env.n_items):
            # Find all active suppliers for item i
            active_supplier_indices = np.where(bsp_low_ew_policy_matrix[i, :, 3] == 1)[0]
            if len(active_supplier_indices) == 0:
                continue

            on_hand_item_i = float(current_inventory_level_total[i])
            outstanding_item_i = float(outstanding_orders_total[i])
            inventory_position_y_t = on_hand_item_i + outstanding_item_i

            # Aggregate (S1, S2, b) across active suppliers
            S2_values = np.array([float(bsp_low_ew_policy_matrix[i, s, 1]) for s in active_supplier_indices])
            total_S2 = float(np.sum(S2_values))
            total_S1 = float(np.sum([bsp_low_ew_policy_matrix[i, s, 0] for s in active_supplier_indices]))

            # Weights for averaging (by S2, or equal if all S2 are zero)
            if total_S2 > 1e-9:
                weights = S2_values / total_S2
            else:
                weights = np.ones(len(active_supplier_indices)) / len(active_supplier_indices)

            avg_b = float(np.sum([
                weights[k] * bsp_low_ew_policy_matrix[i, s, 2]
                for k, s in enumerate(active_supplier_indices)
            ]))

            # EW using weighted average lead time
            avg_lead_time = int(np.round(np.sum([
                weights[k] * self.env.lead_times[i, s]
                for k, s in enumerate(active_supplier_indices)
            ])))
            avg_lead_time = max(1, avg_lead_time)

            ew_item_i = 0.0
            if self.waste_estimation_method == "closed_form_approx":
                ew_item_i = self._calculate_ew_closed_form_approx(i, avg_lead_time)
            elif self.waste_estimation_method == "deterministic_simulation":
                ew_item_i = self._calculate_ew_deterministic_simulation(i, avg_lead_time)
            else:
                if self.env.current_step < 5:
                    print(f"Warning: Unknown EW method '{self.waste_estimation_method}' in BSPELowAgent. Using 0 EW for item {i}.")

            # Compute total order using aggregated BSP-low formula
            total_order = 0.0
            if inventory_position_y_t < avg_b:
                if avg_b < 1e-6:
                    total_order = total_S2 - inventory_position_y_t + ew_item_i
                else:
                    order_at_y0 = total_S1
                    order_at_b = total_S2 - avg_b
                    slope_m = (order_at_b - order_at_y0) / avg_b
                    total_order = order_at_y0 + slope_m * inventory_position_y_t + ew_item_i
            else:
                total_order = total_S2 - inventory_position_y_t + ew_item_i

            total_order = max(0.0, total_order)

            # Split proportionally by S2 values
            if total_order > 0.0 and total_S2 > 1e-9:
                for k, s in enumerate(active_supplier_indices):
                    fraction = S2_values[k] / total_S2
                    action[i, s] = float(fraction * total_order)
            elif total_order > 0.0:
                # Equal split if all S2 are zero
                per_supplier = total_order / len(active_supplier_indices)
                for s in active_supplier_indices:
                    action[i, s] = float(per_supplier)
        return action

    def run(self, render_steps=False, verbose=False):
        all_episode_rewards = []
        if self.optimized_policy is None:
            print("Error: BSPELowAgent has no optimized policy. Check optimization/loading. Exiting.", file=sys.stderr)
            return []
        if np.sum(self.optimized_policy[:,:,3]) == 0 and not self.load_policy_path: # No active suppliers and not loaded
            print("Error: BSPELowAgent optimized policy has no active suppliers. Exiting.", file=sys.stderr)
            # This can happen if num_candidate_policies is too low or options are poor.
            return []


        print(f"\nRunning final evaluation with {'Loaded' if self.load_policy_path else 'Optimized'} BSP-low-EW for {self.num_final_eval_episodes} episode(s)...")
        self._reset_env_seed_sequence()
        for episode_idx in range(self.num_final_eval_episodes):
            if self.logger:
                self.logger.start_episode(episode_num=episode_idx)

            observation, info_reset = self.env.reset()
            terminated = False
            truncated = False
            total_reward_episode = 0.0

            while not (terminated or truncated):
                current_step_env = self.env.current_step
                action = self._get_action_from_policy(self.optimized_policy) # This will use the BSPELowAgent's method
                observation_next, reward_step, terminated, truncated, info_step = self.env.step(action, verbose=False) # Pass verbose from run to step

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
            if verbose: print(f"Evaluation Episode {episode_idx + 1} (BSPELowAgent): Total Reward: {total_reward_episode:.2f}")
            all_episode_rewards.append(total_reward_episode)

        if self.logger: self.logger.finalize_logs()

        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            std_final_reward = np.std(all_episode_rewards)
            print(f"BSPELowAgent: Average reward over {self.num_final_eval_episodes} final evaluation episodes: {avg_final_reward:.2f} +/- {std_final_reward:.2f}")
        else:
            print("BSPELowAgent: No final evaluation episodes run.")
        return all_episode_rewards