import os
import sys
import json
import random
import numpy as np
import torch
from envs import SimplePlant
from src.scenarioManager.stochasticDemandModel import StochasticDemandModel
from src.agents.FixedPolicyAgent import FixedPolicyAgent

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment_name>")
        return

    experiment_name = sys.argv[1]

    # Setting the seeds
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # If you are using CUDA (PyTorch with GPU support)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Environment setup load:
    file_path = os.path.abspath(f"./cfg_env/setting_{experiment_name}.json")
    with open(file_path, 'r') as fp:
        settings = json.load(fp)

    # Models setups:
    stoch_model = StochasticDemandModel(settings)
    settings['time_horizon'] = 100
    env = SimplePlant(settings, stoch_model)
    settings['dict_obs'] = False

    # Example fixed action
    fixed_action = [1, 1]  # Example fixed action
    agent = FixedPolicyAgent(env, fixed_action)
    agent.run(n_episodes=1)

if __name__ == "__main__":
    main()
