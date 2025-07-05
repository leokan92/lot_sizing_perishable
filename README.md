# Optimizing Integrated Procurement and Perishable Inventory Policies with Population-Based Metaheuristics

**Authors:**
*   Leonardo Kanashiro Felizardo (Escola Politécnica da Universidade de São Paulo)
*   Edoardo Fadda (Politecnico di Torino)
*   Mariá Cristina Vasconcelos Nascimento (Instituto Tecnológico de Aeronáutica)

---

## Abstract

Managing perishable inventory presents a formidable challenge in operations research, characterized by the dual risks of spoilage and stockouts. These challenges are compounded in environments with stochastic demand, unreliable suppliers, and multi-item, multi-supplier sourcing complexities, which create high-dimensional, coupled decision problems that are intractable for traditional optimization methods. This paper addresses this gap by developing a discrete-event simulation environment and using it to conduct a benchmarking study of several control strategies. We compare the performance of classic parametrized heuristics—Constant Order Policy (COP), Base Stock Policy (BSP), and an Expected Wastage-Aware variant (BSP-EW)—against a suite of population-based metaheuristics: the Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and the Non-dominated Sorting Genetic Algorithm II (NSGA-II). These metaheuristics are employed to tune parameters and construct control policies by automating the selection of the most suitable heuristic and its parameters on an item-by-item basis. Our computational results, conducted across twelve distinct scenarios of varying scale and uncertainty, demonstrate the superiority of the NSGA-II-driven approach, which achieved the highest total reward in all tested instances. The policies discovered by NSGA-II prove to be most effective in balancing the critical trade-off between minimizing lost sales and controlling inventory-related costs. The primary contribution of this work is the empirical validation that advanced metaheuristics provide a framework for solving complex, integrated procurement and perishable inventory problems, justifying the upfront computational investment with significant improvements in operational efficiency and service levels.

---

## Project Overview

This repository contains the source code and experimental setup for the research paper titled "Optimizing Integrated Procurement and Perishable Inventory Policies with Population-Based Metaheuristics". The project focuses on developing and benchmarking control strategies for complex perishable inventory management problems characterized by stochastic demand, unreliable suppliers, and multi-item, multi-supplier sourcing.

The core of the project is a custom discrete-event simulation environment and a collection of agents implementing various control policies, from simple heuristics to advanced metaheuristic-driven approaches.

---

## Repository Structure

The project is organized to separate the environment, agents, configurations, and results.

```plaintext
├───src
│   ├───agents          # Implementations of COP, BSP, BSP-EW, and Metaheuristic agents
│   ├───envs            # Perishable inventory simulation environment (perishableInvEnv.py)
│   ├───cfg_agent       # JSON configuration files for each agent
│   ├───cfg_env         # JSON configuration files for different experimental scenarios
│   ├───cfg_experiments # CSV files for batch-running experiments
│   └───results         # Directory for output logs and result summaries
└───main_runner.py      # Main script to execute experiment batches
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

---

## Running Experiments

The primary entry point for executing experiments is `main_runner.py`.

1.  **Define Experiments:** Experiments are defined in a CSV file (e.g., `src/cfg_experiments/experiments_batch.csv`). Each row specifies an experimental run, including the environment configuration, agent configuration, number of seeds, and paths for saving/loading policies.
2.  **Configure Environments and Agents:** The specific dynamics of an environment (e.g., number of items, demand uncertainty) and the parameters of an agent (e.g., population size for GA) are defined in JSON files located in `src/cfg_env` and `src/cfg_agent`, respectively.
3.  **Execute:** Run the main script, pointing it to your desired batch file.

    ```bash
    python main_runner.py --batch_file ./src/cfg_experiments/experiments_batch.csv
    ```
4.  **Collect Results:** The script runs each experiment for the specified number of seeds, captures performance metrics (total reward, execution time), and saves an aggregated summary to a CSV file in the `src/results/simulation_logs` directory. Detailed step-by-step logs can also be enabled via agent configuration.

---

## How to Cite

If you use this code or the findings from our study in your research, please cite our paper:

```bibtex
@article{felizardo2025optimizing,
  title={Optimizing Integrated Procurement and Perishable Inventory Policies with Population-Based Metaheuristics},
  author={Felizardo, Leonardo Kanashiro and Fadda, Edoardo and Nascimento, Mari{\'a} Cristina Vasconcelos},
  journal={Working Paper},
  year={2025}
}
```
