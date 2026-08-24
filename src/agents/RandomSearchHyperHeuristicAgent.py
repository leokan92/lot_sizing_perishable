# src/agents/RandomSearchHyperHeuristicAgent.py
"""Budget-matched random (Monte Carlo) search over the item-level hybrid encoding.

This agent exists to answer the question "how much of the hyper-heuristic's gain
comes from the *encoding* (a per-item choice of heuristic type, parameter and
supplier) and how much from the *population-based search* (GA / EGA / PSO)?".

It reuses `InventoryOptimizationProblem` from `PymooMetaHeuristicAgent`
unchanged, so the decision-variable bounds, the chromosome decoding and the
common-random-numbers fitness estimator are byte-identical to the ones GA / EGA /
PSO see. The only thing that differs is the search engine: candidates are drawn
uniformly at random from the same discrete box instead of being evolved.

The evaluation budget is matched per instance against the population-based runs
that are already on disk: for each reference method we read its
`*_optimized_search.json` log, sum `pop_size` over every bracket-and-bisect trial
and multiply by `n_gen`. Aggregating those with `max` guarantees this agent never
evaluates fewer candidate policies than any population-based method did on the
same instance, the adaptive population-size search included.
"""
import json
import math
import os
import re
import sys
import time

import numpy as np

try:
    from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
    from pymoo.core.callback import Callback
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
except ImportError:
    print("FATAL ERROR: pymoo is not installed. Please install it using 'pip install pymoo'", file=sys.stderr)
    sys.exit(1)

from src.agents.PymooMetaHeuristicAgent import (
    HEURISTIC_COP,
    InventoryOptimizationProblem,
    PymooMetaHeuristicAgent,
)

# Reference runs whose budget we match. `{env_index}` is the scenario number
# parsed out of the env name (e.g. "setting_16" -> "16").
DEFAULT_REFERENCE_SEARCH_LOGS = [
    "./src/results/policies/{env_index}_ga_config_optimized_search.json",
    "./src/results/policies/{env_index}_nsga2_config_optimized_search.json",
    "./src/results/policies/{env_index}_pso_config_optimized_search.json",
]
DEFAULT_REFERENCE_N_GEN = 50
DEFAULT_BATCH_SIZE = 100
DEFAULT_SHORTLIST_SIZE = 5
DEFAULT_FALLBACK_BUDGET = 31050


def _label_from_template(template, env_index):
    """'.../16_ga_config_optimized_search.json' -> 'ga_config'."""
    base = os.path.basename(template).format(env_index=env_index)
    base = re.sub(r"^%s_" % re.escape(str(env_index)), "", base)
    return re.sub(r"_optimized_search\.json$", "", base)


def read_search_log_budget(path, n_gen=DEFAULT_REFERENCE_N_GEN):
    """Total candidate evaluations a bracket-and-bisect run consumed.

    Returns None when the file is missing or is not a search log (a couple of
    smoke-test artifacts in src/results/policies/ store a bare list)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"Warning: could not read search log '{path}': {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        return None
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials or not isinstance(trials[0], dict):
        return None
    try:
        return int(sum(int(t["pop_size"]) for t in trials) * int(n_gen))
    except (KeyError, TypeError, ValueError):
        return None


def resolve_budget_for_setting(env_index,
                               reference_search_logs=None,
                               reference_n_gen=DEFAULT_REFERENCE_N_GEN,
                               aggregate="max",
                               scale=1.0):
    """Budget for one scenario, plus the per-method budgets it was derived from.

    Returns `(budget_or_None, {method_label: budget})`."""
    templates = reference_search_logs or DEFAULT_REFERENCE_SEARCH_LOGS
    per_method = {}
    for template in templates:
        path = os.path.abspath(template.format(env_index=env_index))
        budget = read_search_log_budget(path, reference_n_gen)
        if budget is not None:
            per_method[_label_from_template(template, env_index)] = budget
    if not per_method:
        return None, per_method
    values = list(per_method.values())
    if aggregate == "max":
        aggregated = max(values)
    elif aggregate == "min":
        aggregated = min(values)
    elif aggregate == "mean":
        aggregated = sum(values) / len(values)
    else:
        raise ValueError(f"budget_matching.aggregate='{aggregate}' not supported; use max/min/mean.")
    return int(round(aggregated * float(scale))), per_method


class _TopKCallback(Callback):
    """Keep the K best distinct candidates seen across all random batches.

    GA's population-size search re-scores its 5 trial winners on
    `num_final_eval_episodes` and keeps the best; giving random search the same
    K-candidate re-scoring stage keeps the two selection procedures symmetric."""

    def __init__(self, k):
        super().__init__()
        self.k = int(k)
        self.best = []  # list of (F, key) sorted ascending by F (F = -reward)

    def notify(self, algorithm):
        pop = getattr(algorithm, "pop", None)
        if pop is None or len(pop) == 0:
            return
        X = np.rint(np.atleast_2d(np.asarray(pop.get("X"), dtype=float))).astype(int)
        F = np.asarray(pop.get("F"), dtype=float).ravel()
        merged = dict((key, fit) for fit, key in self.best)
        for i in range(X.shape[0]):
            key = tuple(int(v) for v in X[i, :])
            fit = float(F[i])
            if key not in merged or fit < merged[key]:
                merged[key] = fit
        self.best = sorted(((fit, key) for key, fit in merged.items()),
                           key=lambda item: item[0])[:self.k]


class RandomSearchHyperHeuristicAgent(PymooMetaHeuristicAgent):
    """Uniform random search over the same hybrid encoding as the pymoo agent."""

    def __init__(self, env,
                 algorithm_config: dict = None,
                 budget_matching: dict = None,
                 evaluation_budget: int = None,
                 env_name: str = None,
                 **kwargs):
        # These must exist before super().__init__, which triggers optimization.
        self.env_name = env_name
        self.budget_matching = budget_matching or {}
        self.evaluation_budget_override = evaluation_budget
        self.budget_info = {}
        # main_runner always sets experiment_name to
        # "{env_name}_{agent_name}_{agent_type}_seed{seed}", so it is a usable
        # fallback for the scenario index when env_name was not passed through.
        self._experiment_name_hint = (kwargs.get("logger_settings") or {}).get("experiment_name")

        hp_search = kwargs.pop("hyperparameter_search", None) or {}
        if hp_search.get("enabled", False):
            print("Warning: RandomSearchHyperHeuristicAgent ignores 'hyperparameter_search' "
                  "(there is no population size to tune).", file=sys.stderr)

        super().__init__(
            env,
            algorithm_config=algorithm_config or {"name": "RANDOM", "params": {}},
            hyperparameter_search={"enabled": False},
            **kwargs
        )

    # --- budget resolution -------------------------------------------------
    def _resolve_env_index(self):
        for candidate in (self.env_name, self._experiment_name_hint):
            if not candidate:
                continue
            match = re.search(r"(\d+)", str(candidate))
            if match:
                return match.group(1)
        return None

    def _resolve_budget(self):
        """-> (budget, batch_size, n_batches); also fills self.budget_info."""
        cfg = self.budget_matching
        batch_size = max(1, int(cfg.get("batch_size", DEFAULT_BATCH_SIZE)))
        per_method = {}
        budget = None
        source = "explicit"

        if cfg.get("enabled", False):
            env_index = self._resolve_env_index()
            if env_index is None:
                print("Warning: budget matching enabled but the scenario index could not be "
                      "determined from env_name/experiment_name.", file=sys.stderr)
            else:
                budget, per_method = resolve_budget_for_setting(
                    env_index,
                    reference_search_logs=cfg.get("reference_search_logs"),
                    reference_n_gen=cfg.get("reference_n_gen", DEFAULT_REFERENCE_N_GEN),
                    aggregate=cfg.get("aggregate", "max"),
                    scale=cfg.get("scale", 1.0),
                )
                source = "search_logs"
            if budget is None:
                budget = int(cfg.get("fallback_evaluation_budget", DEFAULT_FALLBACK_BUDGET))
                source = "fallback"
                print(f"Warning: no reference search log resolved; falling back to "
                      f"evaluation_budget={budget}.", file=sys.stderr)

        if budget is None:
            budget = int(self.evaluation_budget_override
                         if self.evaluation_budget_override is not None
                         else self.algorithm_config.get("params", {}).get(
                             "evaluation_budget", DEFAULT_FALLBACK_BUDGET))

        if budget < 1:
            raise ValueError(f"Resolved evaluation budget must be >= 1, got {budget}.")

        n_batches = max(1, math.ceil(budget / batch_size))
        self.budget_info = {
            "budget_source": source,
            "env_name": self.env_name,
            "env_index": self._resolve_env_index(),
            "reference_budgets": per_method,
            "reference_n_gen": cfg.get("reference_n_gen", DEFAULT_REFERENCE_N_GEN),
            "aggregate": cfg.get("aggregate", "max"),
            "scale": float(cfg.get("scale", 1.0)),
            "evaluation_budget": int(budget),
            "batch_size": int(batch_size),
            "n_batches": int(n_batches),
            "candidates_evaluated": int(n_batches * batch_size),
            "num_optimize_eval_episodes": int(self.num_optimize_eval_episodes),
            "num_final_eval_episodes": int(self.num_final_eval_episodes),
        }
        return budget, batch_size, n_batches

    # --- search ------------------------------------------------------------
    def _run_optimization(self):
        self._optimize_policy_random()

    def _optimize_policy_random(self):
        budget, batch_size, n_batches = self._resolve_budget()
        shortlist_size = max(1, int(self.budget_matching.get("final_shortlist_size",
                                                             DEFAULT_SHORTLIST_SIZE)))

        print("--- Optimizing Meta-Heuristic Policy with Random (Monte Carlo) Search ---")
        print(f"  - Budget source: {self.budget_info['budget_source']}")
        if self.budget_info["reference_budgets"]:
            print(f"  - Reference budgets: {self.budget_info['reference_budgets']}")
        print(f"  - Evaluation budget: {budget} candidate policies "
              f"({n_batches} batches x {batch_size})")
        print(f"  - Num Eval Episodes/Candidate (Optimization): {self.num_optimize_eval_episodes}")
        print(f"  - Final shortlist size: {shortlist_size}")

        problem = InventoryOptimizationProblem(self)
        algorithm = RandomSearch(n_points_per_iteration=batch_size,
                                 sampling=IntegerRandomSampling())
        callback = _TopKCallback(shortlist_size)

        start_time = time.time()
        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_batches),
            seed=self.env._initial_seed if hasattr(self.env, '_initial_seed') and self.env._initial_seed is not None else None,
            callback=callback,
            verbose=True,
            save_history=False
        )
        search_time_s = time.time() - start_time
        print(f"\nRandom Search finished in {search_time_s:.2f} seconds.")

        shortlist = list(callback.best)
        if not shortlist and res.X is not None:
            X = np.rint(np.atleast_2d(np.asarray(res.X, dtype=float))).astype(int)
            F = np.asarray(res.F, dtype=float).ravel()
            shortlist = [(float(F[0]), tuple(int(v) for v in X[0, :]))]

        if not shortlist:
            print("Random Search Error: no candidate was evaluated. Using a default policy.",
                  file=sys.stderr)
            self.best_chromosome = [(0, HEURISTIC_COP, 0.0) for _ in range(self.n_items)]
            self.budget_info["shortlist"] = []
            self.budget_info["search_time_s"] = search_time_s
            self._save_budget_log()
            return

        # Re-score the shortlist on the final-evaluation episodes, exactly as the
        # GA population-size search re-scores its trial winners.
        rescore_start = time.time()
        entries = []
        best_reward, best_chromosome = float("-inf"), None
        for rank, (fit, key) in enumerate(shortlist, start=1):
            chromosome = problem._decode_individual(np.asarray(key, dtype=float))
            avg_reward = self._evaluate_chromosome_silent(chromosome, self.num_final_eval_episodes)
            print(f"  shortlist #{rank}: opt_avg_reward={-fit:.2f}, "
                  f"final_eval_avg_reward={avg_reward:.2f}")
            entries.append({
                "rank": rank,
                "opt_avg_reward": -float(fit),
                "final_eval_avg_reward": float(avg_reward),
            })
            if avg_reward > best_reward:
                best_reward, best_chromosome = avg_reward, chromosome
        rescore_time_s = time.time() - rescore_start

        self.best_chromosome = best_chromosome
        self.budget_info["shortlist"] = entries
        self.budget_info["winner"] = {
            "rank": int(np.argmax([e["final_eval_avg_reward"] for e in entries])) + 1,
            "final_eval_avg_reward": float(best_reward),
        }
        self.budget_info["search_time_s"] = search_time_s
        self.budget_info["shortlist_rescore_time_s"] = rescore_time_s

        print(f"Optimized Policy (Best Chromosome) Found:\n{self.best_chromosome}")
        print(f"Best final-eval avg reward: {best_reward:.2f}")

        self._save_budget_log()

    def _save_budget_log(self):
        if not self.budget_matching.get("save_budget_log", True):
            return
        if not self.save_policy_path:
            return
        log_path = os.path.splitext(self.save_policy_path)[0] + "_budget.json"
        payload = dict(self.budget_info)
        payload["method"] = "random_search"
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as fh:
                json.dump(payload, fh, indent=4)
            print(f"Budget log saved to: {log_path}")
        except Exception as exc:
            print(f"Error saving budget log to '{log_path}': {exc}", file=sys.stderr)
