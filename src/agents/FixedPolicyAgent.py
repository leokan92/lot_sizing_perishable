# src/agents/FixedPolicyAgent.py
import numpy as np
import sys # Import sys for checking tqdm

class FixedPolicyAgent:
    def __init__(self, env, num_final_eval_episodes: int = 1, **kwargs): # Add num_final_eval_episodes
        """
        Initializes the agent with a fixed action policy.

        Args:
            env (PerishableInvEnv): The environment instance.
            num_final_eval_episodes (int): Number of episodes for FINAL evaluation run.
            **kwargs: Must contain 'policy_definition' or 'fixed_action'.
        """
        self.env = env
        self.num_final_eval_episodes = num_final_eval_episodes # Store it
        self.fixed_action = self._generate_action(**kwargs)

    def _generate_action(self, **kwargs):
        # (Keep this method as before)
        if 'fixed_action' in kwargs:
            action = np.array(kwargs['fixed_action'], dtype=np.float32)
        elif 'policy_definition' in kwargs:
            definition = kwargs['policy_definition']
            n_items = self.env.n_items
            n_suppliers = self.env.n_suppliers
            action = np.zeros((n_items, n_suppliers), dtype=np.float32)
            if definition.get('type') == 'first_available':
                quantity = float(definition.get('quantity', 0))
                for i in range(n_items):
                    for s in range(n_suppliers):
                        if self.env.item_supplier_matrix[i, s]:
                            action[i, s] = quantity
                            break
            else:
                print(f"Warning: Unknown policy def type '{definition.get('type')}'. Zero action.")
        else:
             raise ValueError("FixedPolicyAgent needs 'fixed_action' or 'policy_definition'.")
        if action.shape != self.env.action_space.shape:
            raise ValueError("Fixed action shape mismatch.")
        print(f"Using Fixed Action:\n{action}")
        return action

    # Modify run to use self.num_final_eval_episodes
    def run(self, render_steps=False, verbose=False):
        """Runs the agent using the fixed policy for FINAL evaluation."""
        all_episode_rewards = []
        # Use self.num_final_eval_episodes here
        print(f"\nRunning final evaluation with Fixed Policy for {self.num_final_eval_episodes} episode(s)...")
        for episode in range(self.num_final_eval_episodes):
            observation, info_reset = self.env.reset()
            state = observation
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = self.fixed_action
                observation_next, reward, terminated, truncated, info = self.env.step(action, verbose=verbose)
                if render_steps: self.env.render()
                total_reward += reward
                state = observation_next
            if verbose: print(f"Evaluation Episode {episode + 1}: Total Reward: {total_reward:.2f}")
            all_episode_rewards.append(total_reward)
        # Use self.num_final_eval_episodes here
        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(f"Average reward over {self.num_final_eval_episodes} final evaluation episodes: {avg_final_reward:.2f}")
        return all_episode_rewards # Return list