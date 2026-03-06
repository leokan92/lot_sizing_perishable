"""
Generate sensitivity analysis config files and experiment batch CSV.

One-factor-at-a-time (OFAT) design:
  - population_size:             10 values × {GA, NSGA2, PSO} = 30
  - num_generations (n_gen):     10 values × {GA, NSGA2, PSO} = 30
  - num_optimize_eval_episodes:  10 values × {GA, NSGA2, PSO} = 30
  - crossover_rate:              10 values × {GA, NSGA2}      = 20
  - mutation_rate:               10 values × {GA, NSGA2}      = 20
  Total: 130 experiments, all on setting_1
"""
import json, os, copy

CFG_DIR = os.path.join("src", "cfg_agent")
EXP_DIR = os.path.join("src", "cfg_experiments")
POLICY_DIR = "./src/results/policies/"

# ── Base templates (matching existing defaults) ─────────────────────────────
GA_BASE = {
    "agent_type": "pymoo_meta_heuristic",
    "params": {
        "algorithm_config": {
            "name": "GA",
            "params": {
                "pop_size": 30,
                "n_gen": 50,
                "crossover_rate": 0.8,
                "mutation_rate": 0.15
            }
        },
        "num_optimize_eval_episodes": 30,
        "num_final_eval_episodes": 50,
        "quantity_options": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "base_stock_level_options": [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26],
        "bspew_waste_estimation_method": "deterministic_simulation",
        "bspew_waste_horizon_review_periods": 1,
        "bspew_num_ew_demand_sim_paths": 30,
        "logger_settings": {
            "log_step_details": True,
            "log_dir": "./src/results/simulation_logs/",
            "experiment_name": "ga_sensitivity_run",
            "log_actions": True
        }
    }
}

NSGA2_BASE = {
    "agent_type": "pymoo_meta_heuristic",
    "params": {
        "algorithm_config": {
            "name": "NSGA2",
            "params": {
                "pop_size": 30,
                "n_gen": 50,
                "crossover_rate": 0.8,
                "mutation_rate": 0.15
            }
        },
        "num_optimize_eval_episodes": 30,
        "num_final_eval_episodes": 50,
        "quantity_options": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "base_stock_level_options": [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26],
        "bspew_waste_estimation_method": "deterministic_simulation",
        "bspew_waste_horizon_review_periods": 1,
        "bspew_num_ew_demand_sim_paths": 30,
        "logger_settings": {
            "log_step_details": True,
            "log_dir": "./src/results/simulation_logs/",
            "experiment_name": "nsga2_sensitivity_run",
            "log_actions": True
        }
    }
}

PSO_BASE = {
    "agent_type": "pymoo_meta_heuristic",
    "params": {
        "algorithm_config": {
            "name": "PSO",
            "params": {
                "pop_size": 30,
                "n_gen": 50
            }
        },
        "num_optimize_eval_episodes": 30,
        "num_final_eval_episodes": 50,
        "quantity_options": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "base_stock_level_options": [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26],
        "bspew_waste_estimation_method": "deterministic_simulation",
        "bspew_waste_horizon_review_periods": 1,
        "bspew_num_ew_demand_sim_paths": 30,
        "logger_settings": {
            "log_step_details": True,
            "log_dir": "./src/results/simulation_logs/",
            "experiment_name": "pso_sensitivity_run",
            "log_actions": True
        }
    }
}

# ── Parameter value ranges ──────────────────────────────────────────────────
PARAM_VALUES = {
    "popsize":    [5, 10, 15, 20, 30, 40, 50, 75, 100, 150],
    "ngen":       [10, 20, 30, 40, 50, 75, 100, 150, 200, 300],
    "numeval":    [5, 10, 15, 20, 30, 40, 50, 75, 100, 150],
    "crossover":  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "mutation":   [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00],
}

# Which algorithms get which parameters
ALGO_PARAMS = {
    "ga":    ["popsize", "ngen", "numeval", "crossover", "mutation"],
    "nsga2": ["popsize", "ngen", "numeval", "crossover", "mutation"],
    "pso":   ["popsize", "ngen", "numeval"],
}

ALGO_BASES = {
    "ga":    GA_BASE,
    "nsga2": NSGA2_BASE,
    "pso":   PSO_BASE,
}


def apply_param(cfg, param_key, value):
    """Return a deep copy with exactly one parameter changed from its default."""
    c = copy.deepcopy(cfg)
    algo_params = c["params"]["algorithm_config"]["params"]
    if param_key == "popsize":
        algo_params["pop_size"] = value
    elif param_key == "ngen":
        algo_params["n_gen"] = value
    elif param_key == "numeval":
        c["params"]["num_optimize_eval_episodes"] = value
    elif param_key == "crossover":
        algo_params["crossover_rate"] = value
    elif param_key == "mutation":
        algo_params["mutation_rate"] = value
    return c


def value_label(param_key, value):
    """Short suffix for the file name."""
    if param_key in ("crossover", "mutation"):
        return f"{value:.2f}".replace(".", "")        # 0.15 → "015"
    return str(int(value))


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    csv_rows = []
    csv_header = "env_name;agent_name;start_seed;num_seeds;load_policy_file;save_policy_file;render;verbose"

    created_files = 0
    for algo, params in ALGO_PARAMS.items():
        base = ALGO_BASES[algo]
        for param_key in params:
            for val in PARAM_VALUES[param_key]:
                cfg = apply_param(base, param_key, val)
                label = value_label(param_key, val)
                config_name = f"{algo}_sens_{param_key}_{label}"
                file_path = os.path.join(CFG_DIR, f"{config_name}.json")

                with open(file_path, "w") as f:
                    json.dump(cfg, f, indent=4)
                created_files += 1

                policy_ext = ".json"
                policy_file = f"{POLICY_DIR}1_{config_name}_optimized{policy_ext}"
                csv_rows.append(
                    f"setting_1;{config_name};42;1;;{policy_file};FALSE;FALSE"
                )

    # Write experiment batch CSV
    csv_path = os.path.join(EXP_DIR, "experiments_batch_sensitivity.csv")
    with open(csv_path, "w") as f:
        f.write(csv_header + "\n")
        for row in csv_rows:
            f.write(row + "\n")

    print(f"Created {created_files} config files in {CFG_DIR}/")
    print(f"Created experiment batch with {len(csv_rows)} rows → {csv_path}")


if __name__ == "__main__":
    main()
