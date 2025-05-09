# main_runner.py

import os
import sys
import json
import argparse
import random
import numpy as np
import time
import importlib
import csv
from datetime import datetime

# --- Utility Functions (set_seed, load_config, get_agent_class - keep as before) ---
def set_seed(seed_value):
    """Sets seeds for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
    except ImportError: pass
    print(f"Seeds set to: {seed_value}")

def load_config(file_path):
    """Loads JSON configuration from a file."""
    try:
        with open(file_path, 'r') as fp:
            config = json.load(fp)
        print(f"Loaded configuration from: {file_path}")
        return config
    except FileNotFoundError: print(f"Error: Config file not found: {file_path}", file=sys.stderr); sys.exit(1)
    except json.JSONDecodeError as e: print(f"Error: JSON decode error in {file_path}: {e}", file=sys.stderr); sys.exit(1)

def get_agent_class(agent_type_name):
    """Dynamically imports and returns the agent class."""
    agent_mapping = {
        "fixed": ("FixedPolicyAgent", "src.agents.FixedPolicyAgent"),
        "cop": ("ConstantOrderPolicyAgent", "src.agents.ConstantOrderPolicyAgent"),
        "bsp": ("BaseStockPolicyAgent", "src.agents.BaseStockPolicyAgent")
    }
    if agent_type_name not in agent_mapping:
        raise ValueError(f"Unknown agent type: '{agent_type_name}'")
    class_name, module_path = agent_mapping[agent_type_name]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ModuleNotFoundError:
        print(f"Error: Agent module not found: {module_path}", file=sys.stderr)
        sys.exit(1)
    except AttributeError:
        print(f"Error: Agent class '{class_name}' not found in {module_path}", file=sys.stderr)
        sys.exit(1)
# --- End Utility Functions ---

# --- Main Execution Logic ---
def run_experiment(args, current_seed): # Accept seed
    """Sets up and runs a single experiment for a GIVEN seed."""
    print(f"\n===== Running Experiment for Seed: {current_seed} =====")
    set_seed(current_seed)

    # Load Configs
    env_config_path = os.path.abspath(os.path.join(args.env_config_dir, f"{args.env_name}.json"))
    agent_config_path = os.path.abspath(os.path.join(args.agent_config_dir, f"{args.agent_name}.json"))
    env_settings = load_config(env_config_path)
    agent_config = load_config(agent_config_path)

    # Setup Env
    try:
        from src.scenarioManager.stochasticDemandModel import StochasticDemandModel
        from src.envs.perishableInvEnv import PerishableInvEnv
    except ImportError as e: print(f"Error importing: {e}", file=sys.stderr); sys.exit(1)
    env_settings.setdefault('time_horizon', 10)
    env_settings.setdefault('dict_obs', False)
    try:
        stoch_model = StochasticDemandModel(env_settings)
    except Exception as e:
         print(f"Warning: StochasticDemandModel init failed: {e}. Using placeholder.", file=sys.stderr)
         class PlaceholderDemandModel:
             def __init__(self, s): self.rng = np.random.default_rng(current_seed)
             def generate_scenario(self, n): return np.zeros((s.get('n_items',1), n), dtype=int)
         stoch_model = PlaceholderDemandModel(env_settings)
    env = PerishableInvEnv(env_settings, stoch_model, seed=current_seed)

    # Setup Agent
    agent_type = agent_config.get("agent_type")
    agent_params = agent_config.get("params", {})
    if not agent_type: print(f"Error: 'agent_type' missing in {agent_config_path}", file=sys.stderr); sys.exit(1)

    num_final_eval_episodes = agent_params.get('num_final_eval_episodes', 1) # Default to 1

    AgentClass = get_agent_class(agent_type)
    print(f"Initializing agent '{args.agent_name}' (Type: {agent_type})...")
    start_agent_init = time.time()

    policy_file_path_used = None # Generic variable for the policy file path

    # Check if the agent type is one that might save/load policies (e.g., COP, BSP)
    # You might want to make this check more robust or specific if you add more agent types
    if agent_type in ["cop", "bsp"]: # Add other policy-based agents here
        agent_params['load_policy_path'] = args.load_policy_file # Use generic arg
        if args.load_policy_file:
             policy_file_path_used = args.load_policy_file
             agent_params['save_policy_path'] = None # Don't save if loading
        elif args.save_policy_file: # Use generic arg
             save_dir = os.path.dirname(args.save_policy_file) or '.'
             base_name = os.path.basename(args.save_policy_file)
             name, ext = os.path.splitext(base_name)
             ext = ext or '.npy' # Default to .npy if no extension
             # Append agent type and seed to filename if saving a new policy
             # to avoid overwriting and for clarity
             descriptive_name = f"{name}_{agent_type}"
             save_filename = f"{descriptive_name}_seed{current_seed}{ext}" if args.num_seeds > 1 else f"{descriptive_name}{ext}"
             full_save_path = os.path.join(save_dir, save_filename)
             agent_params['save_policy_path'] = full_save_path
             policy_file_path_used = full_save_path
        else:
             agent_params['save_policy_path'] = None
    else:
        # For agents that don't load/save policies, ensure these params are None
        agent_params['load_policy_path'] = None
        agent_params['save_policy_path'] = None


    agent = AgentClass(env, **agent_params)
    end_agent_init = time.time()
    print(f"Agent initialization/optimization took {end_agent_init - start_agent_init:.2f} seconds.")

    # Run Simulation
    print(f"\n--- Starting Final Evaluation Run ---")
    start_run = time.time()
    episode_rewards = agent.run(render_steps=args.render, verbose=args.verbose)
    end_run = time.time()
    print(f"Simulation run took {end_run - start_run:.2f} seconds.")

    # Calculate Final Metrics
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    std_dev_reward = np.std(episode_rewards) if episode_rewards else 0.0
    min_reward = np.min(episode_rewards) if episode_rewards else 0.0
    max_reward = np.max(episode_rewards) if episode_rewards else 0.0

    # --- Prepare result dictionary, including individual episode rewards ---
    result_data = {
        "Seed": current_seed,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Env_Name": args.env_name,
        "Agent_Name": args.agent_name,
        "Agent_Config": os.path.basename(agent_config_path),
        "Num_Eval_Episodes": len(episode_rewards),
        "Avg_Reward": f"{avg_reward:.4f}",
        "StdDev_Reward": f"{std_dev_reward:.4f}",
        "Min_Reward": f"{min_reward:.4f}",
        "Max_Reward": f"{max_reward:.4f}",
        "Policy_File": policy_file_path_used if policy_file_path_used else "" # Generic key
    }
    for i, ep_reward in enumerate(episode_rewards):
        result_data[f"Reward_Ep{i+1}"] = f"{ep_reward:.4f}"

    env.close()
    print(f"===== Experiment for Seed: {current_seed} Finished =====")
    return result_data
# --- End Main Execution Logic ---


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Run Perishable Inventory Experiments")
    parser.add_argument('--env_name', type=str, help='Environment config name')
    parser.add_argument('--agent_name', type=str, help='Agent config name')
    parser.add_argument('--env_config_dir', type=str, default='./src/cfg_env', help='Env config directory')
    parser.add_argument('--agent_config_dir', type=str, default='./src/cfg_agent', help='Agent config directory')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of different seeds')
    parser.add_argument('--start_seed', type=int, default=int(time.time()) % 10000, help='Starting seed value')
    parser.add_argument('--render', action='store_true', help='Render steps')
    parser.add_argument('--verbose', action='store_true', help='Verbose env.step output')
    # Generic argument names for loading/saving policies
    parser.add_argument('--load_policy_file', type=str, default=None, help='Path to load a pre-trained policy file (.npy)')
    parser.add_argument('--save_policy_file', type=str, default=None, help='Path to save an optimized policy file (.npy)')
    parser.add_argument('--output_file', type=str, default=None, help='Path to CSV for results')

    # Handle Defaults if No Command Line Args
    if len(sys.argv) == 1:
        # --- Default: BSP Agent ---
        print("INFO: No command-line arguments. Using default BSP run.")
        default_output_csv_path = './src/results/exp_bsp_default.csv'
        default_policy_save_path = './src/results/policies/bsp_optimized_default.npy' # File name for the saved policy

        args = parser.parse_args([
           '--env_name', 'setting_1',
           '--agent_name', 'bsp_default',    # Assumes bsp_default.json exists
           '--start_seed', '42',
           '--num_seeds', '1',
           '--output_file', default_output_csv_path,
           '--save_policy_file', default_policy_save_path # Use generic arg name
        ])
        print(f"--> Defaulting to: BSP Agent (bsp_default), saving results to {default_output_csv_path}")
        print(f"--> Optimized BSP policy will be saved to (if not loading): {default_policy_save_path}")

        # --- To make COP the default, uncomment below and comment/remove BSP default ---
        # print("INFO: No command-line arguments. Using default COP run.")
        # default_output_csv_path = './src/results/exp_cop_default.csv'
        # default_policy_save_path = './src/results/policies/cop_optimized_default.npy'
        # args = parser.parse_args([
        #    '--env_name', 'setting_1',
        #    '--agent_name', 'cop_default', # Assumes cop_default.json exists
        #    '--start_seed', '42',
        #    '--num_seeds', '1',
        #    '--output_file', default_output_csv_path,
        #    '--save_policy_file', default_policy_save_path # Use generic arg name
        # ])
        # print(f"--> Defaulting to: COP Agent (cop_default), saving results to {default_output_csv_path}")
        # print(f"--> Optimized COP policy will be saved to (if not loading): {default_policy_save_path}")
        #--------------------------------------------------------------

    else:
        args = parser.parse_args()

    # --- Setup Results Storage & Paths ---
    all_results = []
    if args.output_file:
        abs_output_path = os.path.abspath(args.output_file)
        output_dir = os.path.dirname(abs_output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved/appended to: {abs_output_path}")

    # Handle policy file paths after parsing, using generic arg names
    if args.save_policy_file:
         abs_save_policy_path = os.path.abspath(args.save_policy_file)
         save_dir = os.path.dirname(abs_save_policy_path)
         if save_dir : os.makedirs(save_dir, exist_ok=True)
         # Note: The actual filename generation (with seed, agent type) happens in run_experiment
         # Here, we are just ensuring the directory exists for the user-provided base path.
         args.save_policy_file = abs_save_policy_path # Store the potentially modified absolute path

    if args.load_policy_file:
        abs_load_policy_path = os.path.abspath(args.load_policy_file)
        if not os.path.exists(abs_load_policy_path):
            print(f"Error: Cannot load policy - file not found: {abs_load_policy_path}", file=sys.stderr); sys.exit(1)
        args.load_policy_file = abs_load_policy_path

    # --- Run Experiments for Multiple Seeds ---
    start_all_runs = time.time()
    max_episodes_across_runs = 0
    for i in range(args.num_seeds):
        current_run_seed = args.start_seed + i
        experiment_result = run_experiment(args, current_run_seed)
        if experiment_result:
            all_results.append(experiment_result)
            num_eps_this_run = len([k for k in experiment_result if k.startswith('Reward_Ep')])
            if num_eps_this_run > max_episodes_across_runs:
                max_episodes_across_runs = num_eps_this_run

    end_all_runs = time.time()
    print(f"\n--- All {args.num_seeds} Seed Run(s) Complete ({end_all_runs - start_all_runs:.2f}s) ---")

    # --- Save Aggregated Results ---
    if args.output_file and all_results:
        abs_output_path = os.path.abspath(args.output_file)
        print(f"\nSaving results to {abs_output_path}...")
        file_exists = os.path.exists(abs_output_path)

        header_base = [
            "Seed", "Timestamp", "Env_Name", "Agent_Name", "Agent_Config",
            "Num_Eval_Episodes", "Avg_Reward", "StdDev_Reward", "Min_Reward",
            "Max_Reward", "Policy_File" # Generic header name
        ]
        episode_headers = [f"Reward_Ep{i+1}" for i in range(max_episodes_across_runs)]
        full_header = header_base + episode_headers

        try:
            with open(abs_output_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=full_header, restval='')
                if not file_exists or os.path.getsize(abs_output_path) == 0:
                    writer.writeheader()
                writer.writerows(all_results)
            print("Results saved successfully.")
        except IOError as e:
            print(f"Error writing results to CSV: {e}", file=sys.stderr)
    elif args.output_file:
        print("No results generated to save.")

    print("\nmain_runner.py finished.")