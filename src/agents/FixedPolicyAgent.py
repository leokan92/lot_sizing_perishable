import numpy as np
# Assuming SimplePlant is correctly imported from its location
from src.envs import SimplePlant
# Assuming StochasticDemandModel is correctly imported
# from src.scenarioManager.stochasticDemandModel import StochasticDemandModel

class FixedPolicyAgent:
    def __init__(self, env: SimplePlant, fixed_action):
        """
        Initializes the agent with a fixed action policy.

        Args:
            env (SimplePlant): The inventory management environment instance.
            fixed_action (np.ndarray): The action to take at every step.
                                      Shape should match env.action_space.
        """
        self.env = env
        # Ensure fixed_action is a NumPy array for consistency if needed
        self.fixed_action = np.array(fixed_action)
        # Optional: Validate action shape against env.action_space
        # if self.fixed_action.shape != self.env.action_space.shape:
        #    raise ValueError(f"Fixed action shape {self.fixed_action.shape} does not match env action space {self.env.action_space.shape}")


    def run(self, n_episodes=1, render_steps=False):
        """
        Runs the agent in the environment for a specified number of episodes.

        Args:
            n_episodes (int): The number of episodes to run.
            render_steps (bool): Whether to call env.render() after each step.
        """
        all_episode_rewards = []
        for episode in range(n_episodes):
            # Reset returns observation and info dictionary
            observation, info_reset = self.env.reset()
            # We typically don't need the info from reset in the main loop
            state = observation # Use the observation as the current state

            terminated = False
            truncated = False
            total_reward = 0.0 # Use float for rewards

            # Loop continues until the episode is terminated OR truncated
            while not (terminated or truncated):
                # Use the predefined fixed action
                action = self.fixed_action

                # --- Unpack all 5 return values from step ---
                observation_next, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                # ---------------------------------------------

                total_reward += reward
                state = observation_next # Update state for the next iteration (if needed, though not used here)

                if render_steps:
                    self.env.render() # Call render if requested

            print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")
            all_episode_rewards.append(total_reward)

        return all_episode_rewards # Return list of total rewards per episode