# src/agents/BaseStockPolicyAgent.py
import numpy as np
import time
import sys
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

class BaseStockPolicyAgent:
    def __init__(self, env,
                 num_candidate_policies: int = 100,
                 num_optimize_eval_episodes: int = 10,
                 num_final_eval_episodes: int = 1,
                 base_stock_level_options: list = [5, 10, 15, 20, 25, 30, 40, 50], # Renamed & adjusted param
                 load_policy_path: str = None,
                 save_policy_path: str = None,
                 **other_kwargs):
        """
        Initializes the agent with a Base Stock Policy (BSP).
        Optimizes the BSP (target level + supplier per item) using Monte Carlo simulation
        OR loads a pre-optimized policy.

        Args:
            env (PerishableInvEnv): The environment instance.
            num_candidate_policies (int): Number of different random BSPs to evaluate if not loading.
            num_optimize_eval_episodes (int): Number of episodes per candidate during OPTIMIZATION.
            num_final_eval_episodes (int): Number of episodes for FINAL evaluation run.
            base_stock_level_options (list): List of discrete target stock levels for optimization.
            load_policy_path (str, optional): Path to load policy from (.npy). Skips optimization.
                                              Policy format: (n_items, n_suppliers) matrix where
                                              non-zero entry [i, s] is the base stock level for item i
                                              ordered from supplier s. Only one s per i is non-zero.
            save_policy_path (str, optional): Path to save optimized policy to (.npy).
            **other_kwargs: Allows ignoring extra params passed from config.
        """
        self.env = env
        # Store parameters
        self.num_candidate_policies = num_candidate_policies
        self.num_optimize_eval_episodes = num_optimize_eval_episodes # For optimization phase
        self.num_final_eval_episodes = num_final_eval_episodes       # For final run phase
        self.base_stock_level_options = base_stock_level_options
        self.load_policy_path = load_policy_path
        self.save_policy_path = save_policy_path
        self.optimized_policy = None # Stores the (n_items, n_suppliers) base stock level matrix

        if self.load_policy_path:
            print(f"\n--- Loading Pre-optimized BSP Policy ---")
            try:
                self.optimized_policy = np.load(self.load_policy_path)
                print(f"Policy successfully loaded from: {self.load_policy_path}")
                print(f"Loaded Policy (Base Stock Levels per Item/Supplier):\n{self.optimized_policy}")
                if self.optimized_policy.shape != self.env.action_space.shape:
                     raise ValueError(f"Loaded policy shape {self.optimized_policy.shape} incompatible with env action space {self.env.action_space.shape}")
                # Validation: Check if only one supplier is chosen per item
                for i in range(self.env.n_items):
                    if np.count_nonzero(self.optimized_policy[i, :]) > 1:
                        print(f"Warning: Loaded policy for item {i} has multiple suppliers defined. Using first found.")
                        # Optional: Could enforce strict single supplier here if needed
            except Exception as e:
                print(f"Error loading policy from '{self.load_policy_path}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # --- Perform Optimization ---
            print("Optimizing Base Stock Policy (Single Supplier per Item Variant) using Monte Carlo...")
            print(f"  - Num Candidate Policies: {self.num_candidate_policies}")
            print(f"  - Eval Episodes/Policy (Optimization): {self.num_optimize_eval_episodes}")
            print(f"  - Base Stock Level Options: {self.base_stock_level_options}")

            start_time = time.time()
            self.optimized_policy = self._optimize_bsp()
            end_time = time.time()
            print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")
            print(f"Optimized Base Stock Policy Found (Levels per Item/Supplier):\n{self.optimized_policy}")

            # --- Save Optimized Policy ---
            if self.save_policy_path:
                print(f"\n--- Saving Optimized BSP Policy ---")
                try:
                    os.makedirs(os.path.dirname(self.save_policy_path), exist_ok=True)
                    np.save(self.save_policy_path, self.optimized_policy)
                    print(f"Optimized policy saved to: {self.save_policy_path}")
                except Exception as e:
                    print(f"Error saving policy to '{self.save_policy_path}': {e}", file=sys.stderr)

    def _calculate_outstanding_orders(self) -> np.ndarray:
        """Calculates the total quantity of each item currently outstanding (ordered but not received)."""
        outstanding = np.zeros(self.env.n_items, dtype=int)
        current_time = self.env.current_step
        lead_times = self.env.lead_times
        for t_placed, i, s, qty_ordered in self.env.order_history:
             # Check if the order placed at t_placed for item i from supplier s
             # with lead time lead_times[i, s] is due to arrive AFTER the current time.
             if t_placed + lead_times[i, s] > current_time:
                   outstanding[i] += qty_ordered
        return outstanding

    def _get_action_from_policy(self, bsp_policy_matrix: np.ndarray) -> np.ndarray:
        """Calculates the order action for the current step based on the BSP policy and current state."""
        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        current_inventory_level = self.env.inventory_level # Total on-hand for each item
        outstanding_orders = self._calculate_outstanding_orders()

        for i in range(self.env.n_items):
            # Find the chosen supplier and target level for item i from the policy matrix
            chosen_supplier_idx = np.where(bsp_policy_matrix[i, :] > 0)[0] # Find index of non-zero element

            if len(chosen_supplier_idx) > 0:
                s = chosen_supplier_idx[0] # Take the first one if multiple (shouldn't happen with generation logic)
                target_level = bsp_policy_matrix[i, s]
                on_hand = current_inventory_level[i]
                outstanding = outstanding_orders[i]

                # Calculate order quantity: Order up to target level, considering on-hand and outstanding
                order_qty = max(0, target_level - on_hand - outstanding)
                action[i, s] = float(order_qty)
            # Else: If no supplier has a non-zero base stock level for this item, order nothing (action[i,:] remains 0)

        return action

    def _evaluate_policy(self, policy_matrix: np.ndarray) -> float:
        """Evaluates a given BSP policy matrix over multiple episodes during OPTIMIZATION."""
        total_reward_across_episodes = 0.0
        # Use num_optimize_eval_episodes for evaluating candidates
        for _ in range(self.num_optimize_eval_episodes):
            observation, _ = self.env.reset() # Resets env state including inventory, orders, time
            terminated = False
            truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                # Calculate action based on the policy *and current env state*
                action = self._get_action_from_policy(policy_matrix)
                observation_next, reward, terminated, truncated, info = self.env.step(action, verbose=False)
                episode_reward += reward
                observation = observation_next # Not strictly needed for BSP, but good practice
            total_reward_across_episodes += episode_reward

        # Use num_optimize_eval_episodes for averaging
        if self.num_optimize_eval_episodes == 0: return -np.inf
        return total_reward_across_episodes / self.num_optimize_eval_episodes

    def _generate_random_bsp(self) -> np.ndarray:
        """Generates a random BSP policy matrix (one supplier per item, random base stock level)."""
        n_items = self.env.n_items
        n_suppliers = self.env.n_suppliers
        item_supplier_matrix = self.env.item_supplier_matrix
        candidate_bsp = np.zeros((n_items, n_suppliers), dtype=np.float32)

        for i in range(n_items):
            valid_suppliers_for_i = np.where(item_supplier_matrix[i, :] == 1)[0]
            num_valid_suppliers = len(valid_suppliers_for_i)

            if num_valid_suppliers == 0: continue # No supplier for this item

            # Choose one supplier for this item
            if num_valid_suppliers == 1:
                s_chosen = valid_suppliers_for_i[0]
            else:
                s_chosen = self.env.env_rng.choice(valid_suppliers_for_i) # Use env's RNG for consistency if needed

            # Choose a base stock level for this item
            if not self.base_stock_level_options:
                chosen_level = 0 # Default to 0 if no options provided
            else:
                chosen_level = self.env.env_rng.choice(self.base_stock_level_options)

            candidate_bsp[i, s_chosen] = float(chosen_level)

        return candidate_bsp

    def _optimize_bsp(self) -> np.ndarray:
        """Optimizes the BSP using Monte Carlo simulation."""
        best_avg_reward = -np.inf
        best_bsp_policy = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)

        # Setup progress bar if tqdm is available
        if 'tqdm' in sys.modules and self.num_candidate_policies > 0:
             iterator = tqdm(range(self.num_candidate_policies), desc="Optimizing BSP", unit="policy")
        else:
             iterator = range(self.num_candidate_policies)

        for i in iterator:
            candidate_bsp = self._generate_random_bsp()
            avg_reward = self._evaluate_policy(candidate_bsp)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_bsp_policy = candidate_bsp.copy()
                # Update progress bar postfix
                if 'tqdm' in sys.modules and hasattr(iterator, 'set_postfix'):
                     iterator.set_postfix({"Best Avg Reward": f"{best_avg_reward:.2f}"}, refresh=True)

        if best_avg_reward == -np.inf:
            print("\nWarning: No valid policy improved initial reward. Defaulting to zero base stock levels.")
        else:
            print(f"\nOptimization complete. Best Average Reward found (during optimization): {best_avg_reward:.2f}")

        return best_bsp_policy

    def run(self, render_steps=False, verbose=False):
        """Runs the agent using the optimized/loaded BSP for FINAL evaluation."""
        all_episode_rewards = []
        if self.optimized_policy is None:
             print("Error: Agent has no optimized policy to run. Exiting.", file=sys.stderr)
             return []

        # Use self.num_final_eval_episodes for the final run loop
        print(f"\nRunning final evaluation with {'Loaded' if self.load_policy_path else 'Optimized'} BSP for {self.num_final_eval_episodes} episode(s)...")
        for episode in range(self.num_final_eval_episodes):
            observation, info_reset = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                # Get action based on the *optimized* policy and current state
                action = self._get_action_from_policy(self.optimized_policy)
                observation_next, reward, terminated, truncated, info = self.env.step(action, verbose=verbose)
                if render_steps: self.env.render()
                total_reward += reward
                observation = observation_next # Update observation (state)
            if verbose: print(f"Evaluation Episode {episode + 1}: Total Reward: {total_reward:.2f}")
            all_episode_rewards.append(total_reward)

        # Use self.num_final_eval_episodes for averaging and printing
        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(f"Average reward over {self.num_final_eval_episodes} final evaluation episodes: {avg_final_reward:.2f}")
        return all_episode_rewards # Return list of rewards