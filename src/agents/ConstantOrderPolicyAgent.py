# src/agents/ConstantOrderPolicyAgent.py
import numpy as np
import time
import sys
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

class ConstantOrderPolicyAgent:
    def __init__(self, env,
                 num_candidate_policies: int = 100,
                 num_optimize_eval_episodes: int = 10, # Renamed param
                 num_final_eval_episodes: int = 1,     # New param
                 quantity_options: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 load_policy_path: str = None,
                 save_policy_path: str = None,
                 **other_kwargs):
        """
        Initializes the agent with a Constant Order Policy (COP).
        Optimizes the COP using Monte Carlo simulation OR loads a pre-optimized policy.

        Args:
            env (PerishableInvEnv): The environment instance.
            num_candidate_policies (int): Number of different random COPs to evaluate if not loading.
            num_optimize_eval_episodes (int): Number of episodes per candidate during OPTIMIZATION.
            num_final_eval_episodes (int): Number of episodes for FINAL evaluation run.
            quantity_options (list): List of discrete order quantities for optimization.
            load_policy_path (str, optional): Path to load policy from (.npy). Skips optimization.
            save_policy_path (str, optional): Path to save optimized policy to (.npy).
            **other_kwargs: Allows ignoring extra params passed from config.
        """
        self.env = env
        # Store parameters
        self.num_candidate_policies = num_candidate_policies
        self.num_optimize_eval_episodes = num_optimize_eval_episodes # For optimization phase
        self.num_final_eval_episodes = num_final_eval_episodes       # For final run phase
        self.quantity_options = quantity_options
        self.load_policy_path = load_policy_path
        self.save_policy_path = save_policy_path
        self.optimized_action = None # Initialize

        if self.load_policy_path:
            print(f"\n--- Loading Pre-optimized COP Policy ---")
            try:
                self.optimized_action = np.load(self.load_policy_path)
                print(f"Policy successfully loaded from: {self.load_policy_path}")
                print(f"Loaded Policy:\n{self.optimized_action}")
                if self.optimized_action.shape != self.env.action_space.shape:
                     raise ValueError(f"Loaded policy shape {self.optimized_action.shape} incompatible")
            except Exception as e:
                print(f"Error loading policy from '{self.load_policy_path}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # --- Perform Optimization ---
            print("Optimizing Constant Order Policy (Single Supplier per Item Variant) using Monte Carlo...")
            print(f"  - Num Candidate Policies: {self.num_candidate_policies}")
            print(f"  - Eval Episodes/Policy (Optimization): {self.num_optimize_eval_episodes}") # Clarify which episodes
            print(f"  - Quantity Options: {self.quantity_options}")

            start_time = time.time()
            self.optimized_action = self._optimize_cop()
            end_time = time.time()
            print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")
            print(f"Optimized Constant Order Policy Found:\n{self.optimized_action}")

            # --- Save Optimized Policy ---
            if self.save_policy_path:
                print(f"\n--- Saving Optimized COP Policy ---")
                try:
                    os.makedirs(os.path.dirname(self.save_policy_path), exist_ok=True)
                    np.save(self.save_policy_path, self.optimized_action)
                    print(f"Optimized policy saved to: {self.save_policy_path}")
                except Exception as e:
                    print(f"Error saving policy to '{self.save_policy_path}': {e}", file=sys.stderr)

    def _evaluate_policy(self, policy_action: np.ndarray) -> float:
        """Evaluates a given fixed policy over multiple episodes during OPTIMIZATION."""
        total_reward_across_episodes = 0.0
        # Use num_optimize_eval_episodes for evaluating candidates
        for _ in range(self.num_optimize_eval_episodes):
            observation, _ = self.env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                observation_next, reward, terminated, truncated, info = self.env.step(policy_action, verbose=False)
                episode_reward += reward
                observation = observation_next
            total_reward_across_episodes += episode_reward
        # Use num_optimize_eval_episodes for averaging
        if self.num_optimize_eval_episodes == 0: return -np.inf
        return total_reward_across_episodes / self.num_optimize_eval_episodes

    def _generate_random_cop(self) -> np.ndarray:
        # (Keep this method as before)
        n_items = self.env.n_items
        n_suppliers = self.env.n_suppliers
        item_supplier_matrix = self.env.item_supplier_matrix
        candidate_cop = np.zeros((n_items, n_suppliers), dtype=np.float32)
        for i in range(n_items):
            valid_suppliers_for_i = np.where(item_supplier_matrix[i, :] == 1)[0]
            num_valid_suppliers = len(valid_suppliers_for_i)
            if num_valid_suppliers == 0: continue
            elif num_valid_suppliers == 1: s_chosen = valid_suppliers_for_i[0]
            else: s_chosen = np.random.choice(valid_suppliers_for_i)
            if not self.quantity_options: chosen_quantity = 0
            else: chosen_quantity = np.random.choice(self.quantity_options)
            candidate_cop[i, s_chosen] = float(chosen_quantity)
        return candidate_cop

    def _optimize_cop(self) -> np.ndarray:
        # (Keep this method as before, printing result inside)
        best_avg_reward = -np.inf
        best_cop = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        if 'tqdm' in sys.modules and self.num_candidate_policies > 0:
             iterator = tqdm(range(self.num_candidate_policies), desc="Optimizing COP", unit="policy")
        else: iterator = range(self.num_candidate_policies)
        for i in iterator:
            candidate_cop = self._generate_random_cop()
            avg_reward = self._evaluate_policy(candidate_cop)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_cop = candidate_cop.copy()
                if 'tqdm' in sys.modules and hasattr(iterator, 'set_postfix'):
                     iterator.set_postfix({"Best Avg Reward": f"{best_avg_reward:.2f}"}, refresh=True)
        if best_avg_reward == -np.inf:
            print("\nWarning: No valid policy improved initial reward. Defaulting to zero orders.")
        else:
            print(f"\nOptimization complete. Best Average Reward found (during optimization): {best_avg_reward:.2f}")
        return best_cop

    # Modify run to use self.num_final_eval_episodes
    def run(self, render_steps=False, verbose=False):
        """Runs the agent using the optimized/loaded COP for FINAL evaluation."""
        all_episode_rewards = []
        if self.optimized_action is None:
             print("Error: Agent has no optimized policy to run. Exiting.", file=sys.stderr)
             return []

        # Use self.num_final_eval_episodes for the final run loop
        print(f"\nRunning final evaluation with {'Loaded' if self.load_policy_path else 'Optimized'} COP for {self.num_final_eval_episodes} episode(s)...")
        for episode in range(self.num_final_eval_episodes):
            observation, info_reset = self.env.reset()
            state = observation
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = self.optimized_action
                observation_next, reward, terminated, truncated, info = self.env.step(action, verbose=verbose)
                if render_steps: self.env.render()
                total_reward += reward
                state = observation_next
            if verbose: print(f"Evaluation Episode {episode + 1}: Total Reward: {total_reward:.2f}")
            all_episode_rewards.append(total_reward)

        # Use self.num_final_eval_episodes for averaging and printing
        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(f"Average reward over {self.num_final_eval_episodes} final evaluation episodes: {avg_final_reward:.2f}")
        return all_episode_rewards # Return list of rewards