import os
import sys
import json
import random
import numpy as np
import torch
from src.envs.simplePlant import SimplePlant
from src.scenarioManager.stochasticDemandModel import StochasticDemandModel
from src.agents.FixedPolicyAgent import FixedPolicyAgent

def main():
    #experiment_name = sys.argv[1]
    experiment_name = 'setting_1'

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
    file_path = os.path.abspath(f"./src/cfg_env/{experiment_name}.json")
    with open(file_path, 'r') as fp:
        settings = json.load(fp)

    # Models setups:
    stoch_model = StochasticDemandModel(settings)
    settings['time_horizon'] = 10 # TODO include this in the json file
    env = SimplePlant(settings, stoch_model)
    settings['dict_obs'] = False # TODO include this in the json file

    # Define fixed action
    n_items = settings['n_items']
    n_suppliers = settings['n_suppliers']
    fixed_action = np.zeros((n_items, n_suppliers))
    for i in range(n_items):
        for s in range(n_suppliers):
            if settings['item_supplier_matrix'][i][s]:
                fixed_action[i, s] = 10  # Order 10 units from the first available supplier
                break

    agent = FixedPolicyAgent(env, fixed_action)
    agent.run(n_episodes=1)

if __name__ == "__main__":
    main()