# Automatic Policy Search using Population-Based Hyper-heuristics for the Integrated Procurement and Perishable Inventory Problem

**Authors:**
*   Leonardo Kanashiro Felizardo (Escola Politécnica da Universidade de São Paulo)
*   Edoardo Fadda (Politecnico di Torino)
*   Mariá Cristina Vasconcelos Nascimento (Instituto Tecnológico de Aeronáutica)

---

## Abstract

This paper addresses the problem of managing perishable inventory under multiple sources of uncertainty, including stochastic demand, unreliable supplier fulfillment, and probabilistic product shelf life. 
We develop a discrete-event simulation environment to compare two optimization strategies for this multi-item, multi-supplier problem. 
The first strategy optimizes uniform classic policies (e.g., Constant Order and Base Stock) by tuning their parameters globally, complemented by a direct search to select the best-fitting suppliers for the integrated problem. 
The second approach is a hyper-heuristic approach, driven by metaheuristics such as a Genetic Algorithm (GA) and Particle Swarm Optimization (PSO). 
This framework constructs a composite policy by automating the selection of the heuristic type, its parameters, and the sourcing suppliers on an item-by-item basis. 
Computational results from twelve distinct instances demonstrate that the hyper-heuristic framework consistently identifies superior policies, with GA and EGA exhibiting the best overall performance. 
Our primary contribution is verifying that this item-level policy construction yields significant performance gains over simpler global policies, thereby justifying the associated computational cost.

---

## Project Overview

This repository contains the source code and experimental setup for the research paper titled "Automatic Policy Search using Population-Based Hyper-heuristics for the Integrated Procurement and Perishable Inventory Problem". The project focuses on developing and benchmarking control strategies for complex perishable inventory management problems characterized by stochastic demand, unreliable suppliers, and multi-item, multi-supplier sourcing.

The core of the project is a custom discrete-event simulation environment and a collection of agents implementing various control policies, from simple heuristics to advanced metaheuristic-driven approaches.

---

## Repository Structure

The project is organized to separate the environment, agents, configurations, and results.

```plaintext
├───src
│   ├───agents          # COP, BSP, BSP-EW, Metaheuristic, and Reinforcement Learning agents
│   ├───envs            # Perishable inventory simulation environment (perishableInvEnv.py)
│   ├───cfg_agent       # JSON configuration files for each agent
│   ├───cfg_env         # JSON configuration files for experimental scenarios (settings 0–15)
│   ├───cfg_experiments # CSV files for batch-running experiments
│   ├───models          # Optimization problem instance definitions
│   ├───scenarioManager # Stochastic demand model
│   ├───utils           # Simulation logger
│   └───results         # Output logs, policies, and result summaries
├───m5_analysis         # M5 dataset analysis notebooks and data
├───main_runner.py      # Main script to execute experiment batches
├───requirements.txt    # Python dependencies
└───README.md
```

---

## Core Components

### Simulation Environment (`src/envs/perishableInvEnv.py`)

A custom discrete-event simulation environment built using the `gymnasium` library. It models a multi-item, multi-supplier perishable inventory system with key dynamics including:
*   Stochastic demand processes.
*   Item-specific shelf life and wastage models based on survival probabilities.
*   Supplier-specific lead times, unit costs, fixed order costs, and fulfillment reliability.
*   FIFO (First-In, First-Out) demand satisfaction.
*   Inventory value tracking based on the actual purchase cost of items.

### Implemented Agents (`src/agents/`)

We implement and benchmark several control strategies, each encapsulated in its own agent class.

#### Baseline Heuristics
These agents optimize a single, uniform policy type across all items using Monte Carlo simulation.
*   **`ConstantOrderPolicyAgent.py` (COP):** A simple policy that orders a fixed, optimized quantity at every time step.
*   **`BaseStockPolicyAgent.py` (BSP):** A classic policy that orders up to a target base-stock level.
*   **`BSPEWAgent.py` (BSP-EW):** An enhanced BSP that incorporates an estimation of expected future wastage into its ordering decision to proactively mitigate spoilage costs.

#### Metaheuristic Agent
*   **`PymooMetaHeuristicAgent.py`:** A powerful agent that leverages population-based metaheuristics from the `pymoo` library (GA, PSO, NSGA-II). This agent constructs a hybrid policy by finding the optimal combination of heuristic (COP, BSP, or BSP-EW) and its corresponding parameters for each item individually. This allows it to create highly tailored and effective control strategies.

#### Reinforcement Learning Agents (Stable-Baselines3)
*   **`StableBaselinesAgent.py`:** An RL-based agent built on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) supporting three algorithms:
    *   **PPO** (Proximal Policy Optimization) — on-policy, continuous action space.
    *   **SAC** (Soft Actor-Critic) — off-policy, continuous action space.
    *   **DQN** (Deep Q-Network) — off-policy, discrete action catalog.

    Key features:
    *   **Vectorized training** via `DummyVecEnv` or `SubprocVecEnv` with configurable `n_envs` for parallel environment copies.
    *   **Gymnasium-compatible wrapper** that flattens the Dict observation space into a 1D Box and maps a flat action vector to the environment's `(n_items × n_suppliers)` order matrix.
    *   Continuous actions are floored and clipped to produce integer order quantities.
    *   DQN uses a pre-built discrete action catalog sampled at initialization.
    *   Full support for saving/loading trained policies (`.zip` format).

### Multi-Supplier Ordering

All agents support **multi-supplier ordering per item**. Each item can source from one or more valid suppliers simultaneously:
*   **COP/BSP:** Random 1..N active suppliers per item; quantities split proportionally by base-stock levels.
*   **BSP-EW / BSP-EW-Low:** Proportional split with weighted-average lead time for wastage estimation.
*   **Pymoo Metaheuristic:** Gene encodes "number of suppliers" with equal order splitting.
*   **RL (SB3):** The action space covers all valid (item, supplier) pairs directly.

---

## Running Experiments

The primary entry point for executing experiments is `main_runner.py`.

1.  **Define Experiments:** Experiments are defined in a CSV file (e.g., `src/cfg_experiments/experiments_batch.csv`). Each row specifies an experimental run, including the environment configuration, agent configuration, number of seeds, and paths for saving/loading policies. 

2.  **Configure Environments and Agents:** The specific dynamics of an environment (e.g., number of items, demand uncertainty) and the parameters of an agent (e.g., population size for GA, RL hyperparameters) are defined in JSON files located in `src/cfg_env` and `src/cfg_agent`, respectively.

3.  **Execute:** Run the main script, pointing it to your desired batch file.

    ```bash
    # Run default batch (experiments_batch.csv)
    python main_runner.py

    # Run a specific batch
    python main_runner.py --batch_file ./src/cfg_experiments/experiments_batch.csv
    ```

4.  **Collect Results:** The script runs each experiment for the specified number of seeds, captures performance metrics (total reward, execution time), and saves a timestamped aggregated summary CSV to `src/results/simulation_logs/`. Detailed step-by-step logs can also be enabled via agent configuration.

---

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `gymnasium` | 1.2.3 | RL environment interface |
| `stable-baselines3` | 2.7.1 | PPO, SAC, DQN algorithms |
| `torch` | 2.9.1 | Neural network backend |
| `pymoo` | 0.6.1.6 | GA, PSO, NSGA-II metaheuristics |
| `numpy` | 2.4.0 | Numerical computing |
| `pandas` | 2.3.3 | Data handling and CSV I/O |
| `matplotlib` / `seaborn` | 3.10.8 / 0.13.2 | Plotting and visualization |

---

## How to Cite

If you use this code or the findings from our study in your research, please cite our paper:

```bibtex
@article{Felizardo2025,
  title={Automatic Policy Search using Population-Based Hyper-heuristics for the Integrated Procurement and Perishable Inventory Problem},
  author={Felizardo, Leonardo Kanashiro and Fadda, Edoardo and Nascimento, Mari{\'a} Cristina Vasconcelos},
  journal={Working Paper},
  year={2025}
}
```
