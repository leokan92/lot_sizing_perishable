import numpy as np
from envs import SimplePlant
from src.scenarioManager.stochasticDemandModel import StochasticDemandModel

class FixedPolicyAgent:
    def __init__(self, env: SimplePlant, fixed_action):
        self.env = env
        self.fixed_action = fixed_action

    def run(self, n_episodes=1):
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.fixed_action
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                self.env.render()
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    # Example usage
    settings = {
        'time_horizon': 10,
        'n_items': 3,
        'n_machines': 2,
        'machine_production': [[1, 0, 0], [0, 1, 0]],
        'max_inventory_level': [10, 10, 10],
        'holding_costs': [1, 1, 1],
        'lost_sales_costs': [10, 10, 10],
        'order_costs': [[5, 5, 5], [5, 5, 5]],
        'shelf_life': [5, 5, 5],
        'initial_setup': [1, 1],
        'initial_inventory': [5, 5, 5],
        'dict_obs': False
    }
    stoch_model = StochasticDemandModel()
    env = SimplePlant(settings, stoch_model)
    fixed_action = [1, 1]  # Example fixed action
    agent = FixedPolicyAgent(env, fixed_action)
    agent.run(n_episodes=1)
