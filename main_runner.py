import os
import sys

# Pin BLAS/OMP thread pools to 1 BEFORE numpy is imported. This applies to:
#   - The parent process (so sequential --num_workers 1 runs with 1 BLAS thread too).
#   - All spawn workers (they inherit these env vars on creation).
# Without this, each spawned worker would import numpy with whatever the OS-default
# thread count is and fight other workers for cores — which we measured as a 40%
# per-task slowdown.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import argparse
import random
import numpy as np
import time
import importlib
import pandas as pd # For reading the CSV batch file
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    # Python's hash seed for consistent dict iteration (for some versions)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError: pass
    print(f"Seeds set to: {seed_value}")

def load_config(file_path):
    try:
        with open(file_path, 'r') as fp:
            config = json.load(fp)
        print(f"Loaded configuration from: {file_path}")
        return config
    except FileNotFoundError: print(f"Error: Config file not found: {file_path}", file=sys.stderr); sys.exit(1)
    except json.JSONDecodeError as e: print(f"Error: JSON decode error in {file_path}: {e}", file=sys.stderr); sys.exit(1)

def get_agent_class(agent_type_name):
    agent_mapping = {
        "fixed": ("FixedPolicyAgent", "src.agents.FixedPolicyAgent"),
        "cop": ("ConstantOrderPolicyAgent", "src.agents.ConstantOrderPolicyAgent"),
        "bsp": ("BaseStockPolicyAgent", "src.agents.BaseStockPolicyAgent"),
        "bsp_ew": ("BSPEWAgent", "src.agents.BSPEWAgent"),
        "bsp_ew_low": ("BSPEWLowAgent", "src.agents.BSPEWLowAgent"),
        "pymoo_meta_heuristic": ("PymooMetaHeuristicAgent", "src.agents.PymooMetaHeuristicAgent"),
        "random_search_hyper_heuristic": ("RandomSearchHyperHeuristicAgent", "src.agents.RandomSearchHyperHeuristicAgent"),
        "stable_baselines": ("StableBaselinesAgent", "src.agents.StableBaselinesAgent")
    }
    if agent_type_name not in agent_mapping:
        raise ValueError(f"Unknown agent type: '{agent_type_name}'. Available: {list(agent_mapping.keys())}")
    class_name, module_path = agent_mapping[agent_type_name]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ModuleNotFoundError:
        print(f"Error: Agent module not found: {module_path}. Ensure it's in PYTHONPATH or current structure is correct.", file=sys.stderr); sys.exit(1)
    except AttributeError:
        print(f"Error: Agent class '{class_name}' not found in {module_path}", file=sys.stderr); sys.exit(1)

def run_experiment(exp_config, current_seed, default_config_dirs):
    """Sets up and runs a single experiment for a GIVEN seed, based on exp_config."""
    # ADDED: Capture the start time of the entire experiment run
    execution_timestamp = datetime.now()

    print(f"\n===== Running Experiment for Seed: {current_seed} =====")
    print(f"Timestamp: {execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Env_Name: {exp_config['env_name']}, Agent_Name: {exp_config['agent_name']}")
    set_seed(current_seed)

    # Determine config directories
    env_config_dir = exp_config.get('env_config_dir', default_config_dirs['env'])
    agent_config_dir = exp_config.get('agent_config_dir', default_config_dirs['agent'])

    # Load Configs
    env_config_path = os.path.abspath(os.path.join(env_config_dir, f"{exp_config['env_name']}.json"))
    agent_config_path = os.path.abspath(os.path.join(agent_config_dir, f"{exp_config['agent_name']}.json"))
    env_settings = load_config(env_config_path)
    agent_config = load_config(agent_config_path)

    # Setup Env
    try:
        from src.scenarioManager.stochasticDemandModel import StochasticDemandModel
        from src.envs.perishableInvEnv import PerishableInvEnv
    except ImportError as e: print(f"Error importing environment/model: {e}", file=sys.stderr); sys.exit(1)

    env_settings.setdefault('time_horizon', 50) # Default if not specified
    env_settings.setdefault('dict_obs', False)

    try:
        stoch_model = StochasticDemandModel(env_settings)
    except Exception as e:
         print(f"Warning: StochasticDemandModel init failed: {e}. Using placeholder demand model.", file=sys.stderr)
         class PlaceholderDemandModel:
             def __init__(self, settings_dict):
                 self.rng = np.random.default_rng(settings_dict.get('seed', None)) # Use seed from settings
                 self.n_items = settings_dict.get('n_items', 1)
                 self.mean_demands = settings_dict.get('mean_demands_per_item', np.ones(self.n_items)) # Store mean demands
             def generate_scenario(self, n_time_steps):
                 scenario = np.zeros((self.n_items, n_time_steps), dtype=int)
                 for i in range(self.n_items):
                    scenario[i,:] = self.rng.poisson(self.mean_demands[i], size=n_time_steps)
                 return scenario
         stoch_model = PlaceholderDemandModel(env_settings)

    env = PerishableInvEnv(env_settings, stoch_model, seed=current_seed)

    # Setup Agent
    agent_type = agent_config.get("agent_type")
    agent_params = agent_config.get("params", {})
    if not agent_type: print(f"Error: 'agent_type' missing in {agent_config_path}", file=sys.stderr); sys.exit(1)
    print(f"Agent_Type: {agent_type}, Agent_Config_File: {os.path.basename(agent_config_path)}")

    AgentClass = get_agent_class(agent_type)
    print(f"Initializing agent '{exp_config['agent_name']}' (Type: {agent_type})...")

    policy_file_path_used_for_run = None

    load_policy_path_from_csv = exp_config.get('load_policy_file')
    if pd.isna(load_policy_path_from_csv) or not str(load_policy_path_from_csv).strip():
        load_policy_path_from_csv = None

    save_policy_path_from_csv = exp_config.get('save_policy_file')
    if pd.isna(save_policy_path_from_csv) or not str(save_policy_path_from_csv).strip():
        save_policy_path_from_csv = None

    policy_handling_agents = ["cop", "bsp", "bsp_ew", "bsp_ew_low", "pymoo_meta_heuristic",
                              "random_search_hyper_heuristic", "stable_baselines"]
    json_policy_agents = ("pymoo_meta_heuristic", "random_search_hyper_heuristic")
    if agent_type in policy_handling_agents:
        agent_params['load_policy_path'] = load_policy_path_from_csv
        if load_policy_path_from_csv:
            abs_load_path = os.path.abspath(load_policy_path_from_csv)
            if not os.path.exists(abs_load_path):
                print(f"Error: Cannot load policy - file not found: {abs_load_path}. Agent will optimize.", file=sys.stderr)
                agent_params['load_policy_path'] = None
            else:
                agent_params['load_policy_path'] = abs_load_path
            policy_file_path_used_for_run = agent_params['load_policy_path']

            if not save_policy_path_from_csv or save_policy_path_from_csv == load_policy_path_from_csv :
                 agent_params['save_policy_path'] = None
            else:
                abs_save_path = os.path.abspath(save_policy_path_from_csv)
                save_dir = os.path.dirname(abs_save_path)
                if save_dir: os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.basename(abs_save_path)
                name, ext = os.path.splitext(base_name)
                
                if not ext:
                    if agent_type in json_policy_agents: ext = '.json'
                    elif agent_type == "stable_baselines": ext = '.zip'
                    else: ext = '.npy'
                
                if exp_config.get('num_seeds', 1) > 1:
                    save_filename = f"{name}_seed{current_seed}{ext}"
                else:
                    save_filename = f"{name}{ext}"
                agent_params['save_policy_path'] = os.path.join(save_dir, save_filename)
                policy_file_path_used_for_run = agent_params['save_policy_path']

        elif save_policy_path_from_csv:
            abs_save_path = os.path.abspath(save_policy_path_from_csv)
            save_dir = os.path.dirname(abs_save_path)
            if save_dir: os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.basename(abs_save_path)
            name, ext = os.path.splitext(base_name)
            
            if not ext:
                if agent_type in json_policy_agents: ext = '.json'
                elif agent_type == "stable_baselines": ext = '.zip'
                else: ext = '.npy'
            
            if exp_config.get('num_seeds', 1) > 1:
                save_filename = f"{name}_seed{current_seed}{ext}"
            else:
                save_filename = f"{name}{ext}"

            final_save_path = os.path.join(save_dir, save_filename)
            agent_params['save_policy_path'] = final_save_path
            policy_file_path_used_for_run = final_save_path
        else:
            agent_params['save_policy_path'] = None
    else:
        agent_params['load_policy_path'] = None
        agent_params['save_policy_path'] = None

    if policy_file_path_used_for_run:
        print(f"Policy file associated with this run: {policy_file_path_used_for_run}")

    # The budget-matched random search resolves its evaluation budget from the
    # GA/EGA/PSO search logs of the SAME scenario, so it needs the scenario name.
    if agent_type == "random_search_hyper_heuristic":
        agent_params["env_name"] = exp_config['env_name']

    if "logger_settings" not in agent_params: agent_params["logger_settings"] = {}
    experiment_log_name = f"{exp_config['env_name']}_{exp_config['agent_name']}_{agent_type}_seed{current_seed}"
    agent_params["logger_settings"]["experiment_name"] = experiment_log_name
    if "log_dir" not in agent_params["logger_settings"]:
        project_root = os.getcwd()
        default_log_dir = os.path.join(project_root, "src", "results", "simulation_logs")
        agent_params["logger_settings"]["log_dir"] = default_log_dir
    else:
        if not os.path.isabs(agent_params["logger_settings"]["log_dir"]):
            project_root = os.getcwd()
            agent_params["logger_settings"]["log_dir"] = os.path.abspath(os.path.join(project_root, agent_params["logger_settings"]["log_dir"]))


    start_agent_init = time.time()
    agent = AgentClass(env, **agent_params)
    end_agent_init = time.time()
    init_and_train_time = end_agent_init - start_agent_init
    print(f"Agent initialization/optimization took {init_and_train_time:.2f} seconds.")

    print(f"\n--- Starting Final Evaluation Run ---")
    start_run = time.time()

    render_steps = str(exp_config.get('render', 'FALSE')).upper() in ['TRUE', '1', 'T']
    verbose_steps = str(exp_config.get('verbose', 'FALSE')).upper() in ['TRUE', '1', 'T']

    episode_rewards = agent.run(render_steps=render_steps, verbose=verbose_steps)
    end_run = time.time()
    evaluation_time = end_run - start_run
    print(f"Simulation run took {evaluation_time:.2f} seconds.")

    avg_reward, std_dev_reward, min_r, max_r = -np.inf, 0, -np.inf, -np.inf
    if episode_rewards and len(episode_rewards) > 0 :
        avg_reward, std_dev_reward = np.mean(episode_rewards), np.std(episode_rewards)
        min_r, max_r = np.min(episode_rewards), np.max(episode_rewards)
        print(f"Run Summary (Seed {current_seed}): Episodes: {len(episode_rewards)}, "
              f"Avg: {avg_reward:.2f}, StdDev: {std_dev_reward:.2f}, "
              f"Min: {min_r:.2f}, Max: {max_r:.2f}")
    else: print(f"Run Summary (Seed {current_seed}): No episode rewards reported or empty list.")

    env.close()
    print(f"===== Experiment for Seed: {current_seed} Finished =====")
    
    # MODIFIED: Add execution times and timestamp to the returned dictionary
    return {
        'execution_timestamp': execution_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'env_name': exp_config['env_name'],
        'agent_name': exp_config['agent_name'],
        'agent_type': agent_type,
        'seed': current_seed,
        'init_train_time_s': init_and_train_time,
        'evaluation_time_s': evaluation_time,
        'avg_reward': avg_reward,
        'std_reward': std_dev_reward,
        'min_reward': min_r,
        'max_reward': max_r,
        'num_episodes': len(episode_rewards) if episode_rewards else 0,
        'policy_file': policy_file_path_used_for_run if policy_file_path_used_for_run else "N/A (Optimized or Fixed)"
    }


def _worker_run(payload):
    """Worker entry point: pin thread pools to 1, then call run_experiment.

    Must be top-level (picklable) for Windows spawn-based ProcessPoolExecutor.
    """
    exp_row_dict, current_seed, default_config_dirs = payload
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    return run_experiment(exp_row_dict, current_seed, default_config_dirs)


def _start_resource_monitor(out_path, interval=2.0):
    """Sample RSS/CPU%/threads for the parent and all descendants every `interval` s.

    Returns a (stop_event, thread) pair; call stop_event.set() to terminate.
    No-op (returns None, None) if psutil is unavailable.
    """
    try:
        import psutil
    except ImportError:
        print("psutil not installed; skipping resource monitor. `pip install psutil` to enable.", file=sys.stderr)
        return None, None
    import threading, csv
    parent = psutil.Process(os.getpid())
    stop = threading.Event()

    def loop():
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        # Persistent pid -> Process cache. psutil.cpu_percent() returns 0.0 on
        # the FIRST call for a given Process object and only yields a real value
        # on subsequent calls (it measures CPU between calls). Re-creating
        # Process objects each tick would always return 0 — so we cache them.
        proc_cache = {parent.pid: parent}
        try: parent.cpu_percent(interval=None)
        except Exception: pass
        with open(out_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['t', 'pid', 'role', 'rss_mb', 'cpu_pct', 'num_threads'])
            while not stop.is_set():
                now = time.time()
                # Refresh the descendant list, but reuse cached Process objects.
                try:
                    current_descendants = parent.children(recursive=True)
                except psutil.Error:
                    current_descendants = []
                live_pids = {parent.pid}
                for d in current_descendants:
                    live_pids.add(d.pid)
                    if d.pid not in proc_cache:
                        proc_cache[d.pid] = d
                        # Prime so the next sample returns a real value.
                        try: d.cpu_percent(interval=None)
                        except Exception: pass
                # Drop entries for pids that no longer exist.
                for stale_pid in list(proc_cache.keys()):
                    if stale_pid not in live_pids:
                        proc_cache.pop(stale_pid, None)
                for pid, proc in list(proc_cache.items()):
                    try:
                        role = 'parent' if pid == parent.pid else 'worker'
                        writer.writerow([
                            f"{now:.1f}", pid, role,
                            f"{proc.memory_info().rss / 1e6:.1f}",
                            f"{proc.cpu_percent(interval=None):.1f}",
                            proc.num_threads(),
                        ])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        proc_cache.pop(pid, None)
                        continue
                fh.flush()
                stop.wait(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop, t


def _summarize_resource_log(log_path):
    """Read the resource log and print peak RSS, mean CPU% per worker PID."""
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Could not read resource log {log_path}: {e}", file=sys.stderr)
        return
    if df.empty:
        print(f"Resource log {log_path} is empty.")
        return
    workers = df[df['role'] == 'worker']
    parent = df[df['role'] == 'parent']
    print(f"\n--- Resource summary ({log_path}) ---")
    if not parent.empty:
        print(f"Parent peak RSS: {parent['rss_mb'].max():.1f} MB | mean CPU%: {parent['cpu_pct'].mean():.1f}")
    if not workers.empty:
        agg = workers.groupby('pid').agg(peak_rss_mb=('rss_mb', 'max'),
                                          mean_cpu_pct=('cpu_pct', 'mean'),
                                          max_threads=('num_threads', 'max'),
                                          samples=('t', 'count'))
        print(f"Workers ({len(agg)} processes):")
        print(agg.to_string())
        print(f"Total peak RSS across workers (sum of per-worker peaks): {agg['peak_rss_mb'].sum():.1f} MB")
    else:
        print("No worker rows recorded.")


if __name__ == "__main__":
    # ADDED: Define default paths at the top level for consistency
    project_root = os.getcwd()
    default_log_dir = os.path.join(project_root, "src", "results", "simulation_logs")
    default_results_filename = f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    default_results_path = os.path.join(default_log_dir, default_results_filename)

    parser = argparse.ArgumentParser(description="Run Perishable Inventory Experiment Batches from CSV")
    parser.add_argument(
        '--batch_file',
        type=str,
        default='./src/cfg_experiments/experiments_batch.csv',
        help='Path to the CSV file defining experiment batches.'
    )
    parser.add_argument(
        '--default_env_config_dir',
        type=str,
        default='./src/cfg_env',
        help='Default directory for environment JSON configurations.'
    )
    parser.add_argument(
        '--default_agent_config_dir',
        type=str,
        default='./src/cfg_agent',
        help='Default directory for agent JSON configurations.'
    )
    # MODIFIED: Changed default from None to the new timestamped path.
    # Now, by default, it will save a summary CSV with timings in the simulation log directory.
    parser.add_argument(
        '--results_output_csv',
        type=str,
        default=default_results_path,
        help=f'Optional path to save aggregated experiment results to a CSV file. '
             f'Defaults to a timestamped file in: {default_log_dir}'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel experiment-row worker processes. 1 = sequential (default).'
    )
    parser.add_argument(
        '--resource_log',
        type=str,
        default=None,
        help='Optional CSV path. If set, sample RSS/CPU%% of parent + workers every 2s and write here.'
    )
    cli_args = parser.parse_args()

    default_config_dirs = {
        'env': os.path.abspath(cli_args.default_env_config_dir),
        'agent': os.path.abspath(cli_args.default_agent_config_dir)
    }

    batch_file_path = os.path.abspath(cli_args.batch_file)
    if not os.path.exists(batch_file_path):
        print(f"Error: Batch file not found: {batch_file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading experiment batch from: {batch_file_path}")
    try:
        experiments_df = pd.read_csv(batch_file_path, sep = ';')
    except Exception as e:
        print(f"Error reading batch CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    for col in ['start_seed', 'num_seeds']:
        if col in experiments_df.columns:
            experiments_df[col] = pd.to_numeric(experiments_df[col], errors='coerce').fillna(0).astype(int)

    for col in ['env_name', 'agent_name', 'load_policy_file', 'save_policy_file', 'env_config_dir', 'agent_config_dir']:
         if col in experiments_df.columns:
            experiments_df[col] = experiments_df[col].astype(str).str.strip().replace({'nan': None, 'None': None, '': None, 'nan': pd.NA})


    print(f"Found {len(experiments_df)} experiment configurations in batch file.")

    # Build the flat task list: one entry per (exp_row, seed) pair.
    tasks = []
    for index, exp_row in experiments_df.iterrows():
        start_seed = int(exp_row.get('start_seed', 0))
        num_seeds = int(exp_row.get('num_seeds', 1))
        for i in range(num_seeds):
            tasks.append((exp_row.to_dict(), start_seed + i, default_config_dirs))

    print(f"Dispatching {len(tasks)} experiment runs across {max(1, cli_args.num_workers)} worker(s)...")

    monitor_stop, monitor_thread = (None, None)
    if cli_args.resource_log:
        monitor_stop, monitor_thread = _start_resource_monitor(cli_args.resource_log, interval=2.0)

    total_campaigns_start_time = time.time()
    all_results = []

    if cli_args.num_workers <= 1:
        for payload in tasks:
            exp_row_dict, current_run_seed, _ = payload
            print(f"\n\n--- Run: Env='{exp_row_dict.get('env_name')}', Agent='{exp_row_dict.get('agent_name')}', Seed={current_run_seed} ---")
            exp_result = _worker_run(payload)
            if exp_result:
                all_results.append(exp_result)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cli_args.num_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_worker_run, p): p for p in tasks}
            completed = 0
            for fut in as_completed(futures):
                payload = futures[fut]
                exp_row_dict, current_run_seed, _ = payload
                try:
                    exp_result = fut.result()
                    if exp_result:
                        all_results.append(exp_result)
                except Exception as e:
                    print(f"Worker failed for Env='{exp_row_dict.get('env_name')}', "
                          f"Agent='{exp_row_dict.get('agent_name')}', Seed={current_run_seed}: {e}",
                          file=sys.stderr)
                completed += 1
                print(f"[{completed}/{len(tasks)}] finished: "
                      f"Env='{exp_row_dict.get('env_name')}', "
                      f"Agent='{exp_row_dict.get('agent_name')}', "
                      f"Seed={current_run_seed}")

    total_campaigns_end_time = time.time()

    if monitor_stop is not None:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        _summarize_resource_log(cli_args.resource_log)

    print(f"\n--- All Experiment Campaigns Complete ({total_campaigns_end_time - total_campaigns_start_time:.2f}s) ---")
    print(f"Workers: {cli_args.num_workers} | Runs: {len(tasks)} | Wall-clock: {total_campaigns_end_time - total_campaigns_start_time:.2f}s")

    # Stable ordering of results so the CSV is deterministic w.r.t. inputs.
    if all_results:
        all_results.sort(key=lambda r: (str(r.get('env_name')), str(r.get('agent_name')), int(r.get('seed', 0))))
    if cli_args.results_output_csv and all_results:
        results_df = pd.DataFrame(all_results)

        for col in ['init_train_time_s', 'evaluation_time_s']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}")

        try:
            output_path = os.path.abspath(cli_args.results_output_csv)
            # Ensure the directory for the output file exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"\nAggregated experiment results and timings saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving results to CSV {cli_args.results_output_csv}: {e}", file=sys.stderr)
    elif cli_args.results_output_csv:
        print("\nNo results to save to CSV (all_results list is empty).")


    print(f"\nDetailed simulation logs (if enabled in agent's JSON config) are saved in the agent's configured log directory.")
    print("main_runner.py finished.")