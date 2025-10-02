# src/agents/ConstantOrderPolicyAgent.py
import numpy as np
import time
import sys
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

# NEW import for SimulationLogger
try:
    from src.utils.simulation_logger import SimulationLogger
except ImportError:
    print("Warning: SimulationLogger utility not found. Detailed step logging will be disabled for COP.")
    SimulationLogger = None # Define as None if import fails


class ConstantOrderPolicyAgent:
    def __init__(self, env,
                 num_candidate_policies: int = 100,
                 num_optimize_eval_episodes: int = 10,
                 num_final_eval_episodes: int = 1,
                 quantity_options: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 load_policy_path: str = None,
                 save_policy_path: str = None,
                 logger_settings: dict = None, # NEW parameter for logger config
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
            logger_settings (dict, optional): Configuration for SimulationLogger.
            **other_kwargs: Allows ignoring extra params passed from config.
        """
        self.env = env
        self.num_candidate_policies = num_candidate_policies
        self.num_optimize_eval_episodes = num_optimize_eval_episodes
        self.num_final_eval_episodes = num_final_eval_episodes
        self.quantity_options = quantity_options
        self.load_policy_path = load_policy_path
        self.save_policy_path = save_policy_path
        self.optimized_action = None

        # --- Initialize Simulation Logger ---
        self.logger = None
        if SimulationLogger and logger_settings and logger_settings.get("log_step_details", False):
            exp_name_for_logger = logger_settings.get("experiment_name", "cop_default_experiment_unknown_seed")
            
            self.logger = SimulationLogger(
                log_dir=logger_settings.get("log_dir", "./src/results/simulation_logs"),
                experiment_name=exp_name_for_logger,
                log_step_details=logger_settings.get("log_step_details", True),
                log_actions=logger_settings.get("log_actions", True), # COP always knows its action
                n_items=self.env.n_items,
                n_suppliers=self.env.n_suppliers
            )
            if self.logger.log_step_details_enabled:
                 print(f"COP Agent: Detailed simulation logging enabled. Log file: {self.logger.log_file_path}")


        if self.load_policy_path:
            print(f"\n--- Loading Pre-optimized COP Policy ---")
            try:
                self.optimized_action = np.load(self.load_policy_path)
                print(f"Policy successfully loaded from: {self.load_policy_path}")
                print(f"Loaded Policy (Constant Order Quantities):\n{self.optimized_action}")
                if self.optimized_action.shape != self.env.action_space.shape:
                     raise ValueError(f"Loaded COP policy shape {self.optimized_action.shape} "
                                      f"incompatible with env action space {self.env.action_space.shape}")
            except Exception as e:
                print(f"Error loading COP policy from '{self.load_policy_path}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Optimizing Constant Order Policy (Single Supplier per Item Variant) using Monte Carlo...")
            print(f"  - Num Candidate Policies: {self.num_candidate_policies}")
            print(f"  - Eval Episodes/Policy (Optimization): {self.num_optimize_eval_episodes}")
            print(f"  - Quantity Options: {self.quantity_options}")

            start_time = time.time()
            self.optimized_action = self._optimize_cop()
            end_time = time.time()
            print(f"\nCOP Optimization finished in {end_time - start_time:.2f} seconds.")
            print(f"Optimized Constant Order Policy Found:\n{self.optimized_action}")

            if self.save_policy_path:
                print(f"\n--- Saving Optimized COP Policy ---")
                try:
                    os.makedirs(os.path.dirname(self.save_policy_path), exist_ok=True)
                    np.save(self.save_policy_path, self.optimized_action)
                    print(f"Optimized COP policy saved to: {self.save_policy_path}")
                except Exception as e:
                    print(f"Error saving COP policy to '{self.save_policy_path}': {e}", file=sys.stderr)

    def _evaluate_policy(self, policy_action: np.ndarray, seed_batch_key: int | None = None) -> float:
        """Evaluates a given fixed policy over multiple episodes during OPTIMIZATION."""
        total_reward_across_episodes = 0.0
        # CRN: shared seeds per evaluation call
        base = getattr(self.env, '_initial_seed', None)
        base = int(base) if base is not None else 0
        mix_key = int(seed_batch_key) if seed_batch_key is not None else 0
        mixed = (base ^ ((mix_key * 0x9E3779B1) & 0x7FFFFFFF) ^ 0x00C0FFEE) & 0x7FFFFFFF
        rng_local = np.random.default_rng(mixed)
        eval_seeds = [int(s) for s in rng_local.integers(low=0, high=2**31 - 1, size=self.num_optimize_eval_episodes)]

        for seed in eval_seeds:
            observation, _ = self.env.reset(seed=int(seed))
            terminated = False
            truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                observation_next, reward, terminated, truncated, info = self.env.step(policy_action, verbose=False)
                episode_reward += reward
                observation = observation_next
            total_reward_across_episodes += episode_reward
        if self.num_optimize_eval_episodes == 0: return -np.inf
        return total_reward_across_episodes / self.num_optimize_eval_episodes

    def _generate_random_cop(self) -> np.ndarray:
        n_items = self.env.n_items
        n_suppliers = self.env.n_suppliers
        item_supplier_matrix = self.env.item_supplier_matrix
        candidate_cop = np.zeros((n_items, n_suppliers), dtype=np.float32)
        
        # Use env's RNG if available for consistency with BSP and other parts
        rng_to_use = self.env.env_rng if hasattr(self.env, 'env_rng') else np.random.default_rng()

        for i in range(n_items):
            valid_suppliers_for_i = np.where(item_supplier_matrix[i, :] == 1)[0]
            num_valid_suppliers = len(valid_suppliers_for_i)

            if num_valid_suppliers == 0: continue
            elif num_valid_suppliers == 1: s_chosen = valid_suppliers_for_i[0]
            else: s_chosen = rng_to_use.choice(valid_suppliers_for_i)
            
            if not self.quantity_options: chosen_quantity = 0.0
            else: chosen_quantity = rng_to_use.choice(self.quantity_options)
            
            candidate_cop[i, s_chosen] = float(chosen_quantity)
        return candidate_cop

    def _optimize_cop(self) -> np.ndarray:
        best_avg_reward = -np.inf
        best_cop = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        
        if 'tqdm' in sys.modules and self.num_candidate_policies > 0:
             iterator = tqdm(range(self.num_candidate_policies), desc="Optimizing COP", unit="policy")
        else: iterator = range(self.num_candidate_policies)
        
        for iter_idx, _ in enumerate(iterator):
            candidate_cop = self._generate_random_cop()
            avg_reward = self._evaluate_policy(candidate_cop, seed_batch_key=iter_idx)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_cop = candidate_cop.copy()
                if 'tqdm' in sys.modules and hasattr(iterator, 'set_postfix'):
                     iterator.set_postfix({"Best Avg Reward": f"{best_avg_reward:.2f}"}, refresh=True)
        
        if best_avg_reward == -np.inf:
            print("\nWarning: COP - No valid policy improved initial reward. Defaulting to zero orders.")
        else:
            print(f"\nCOP Optimization complete. Best Average Reward found (opt): {best_avg_reward:.2f}")
        return best_cop

    def run(self, render_steps=False, verbose=False):
        """Runs the agent using the optimized/loaded COP for FINAL evaluation."""
        all_episode_rewards = []
        if self.optimized_action is None:
             print("Error: COP Agent has no optimized policy to run. Exiting.", file=sys.stderr)
             return []

        print(f"\nRunning final evaluation with {'Loaded' if self.load_policy_path else 'Optimized'} COP "
              f"for {self.num_final_eval_episodes} episode(s)...")
        
        # The action is constant for COP
        action_to_take = self.optimized_action

        for episode_idx in range(self.num_final_eval_episodes):
            if self.logger:
                self.logger.start_episode(episode_num=episode_idx)

            observation, info_reset = self.env.reset()
            # state = observation # Not strictly needed for COP as action is fixed
            terminated = False
            truncated = False
            total_reward_episode = 0.0
            
            while not (terminated or truncated):
                current_step_env = self.env.current_step # Step number about to be taken

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
                # state = observation_next # Update state (not used for COP's action selection but good practice)
            
            if self.logger:
                self.logger.end_episode()
            
            if verbose: print(f"Evaluation Episode {episode_idx + 1}: Total Reward: {total_reward_episode:.2f}")
            all_episode_rewards.append(total_reward_episode)

        if self.logger:
            self.logger.finalize_logs()

        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(f"Average reward over {self.num_final_eval_episodes} final evaluation episodes (COP): {avg_final_reward:.2f}")
        return all_episode_rewards