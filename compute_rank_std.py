"""Compute mean and standard deviation of each method's rank across the 50
instances, reproducing the average ranks reported in the paper.

Ranking per instance: methods ranked by Avg. Total Reward descending
(less negative = better cost), with ties broken by method order
(pandas rank method='first'). This reproduces the published average ranks
(EGA 1.26, GA 1.80, PSO 3.12, ...).
"""
import os
import numpy as np
import pandas as pd

RUN = os.path.join(os.path.dirname(__file__), 'src', 'analysis',
                   'analysis_run_20260524_193845', 'summary_metrics_table.csv')

# Order used for deterministic tie-breaking (matches the source CSV row order).
METHODS = ['BSP', 'BSP-EW', 'COP', 'DQN', 'EGA', 'GA', 'PPO', 'PSO', 'SAC']


def main():
    df = pd.read_csv(RUN)
    df = df[df['Method'].isin(METHODS)]
    rank_lists = {m: [] for m in METHODS}
    for _setting, sub in df.groupby('Setting'):
        r = (-sub.set_index('Method')['Avg. Total Reward']).rank(method='first')
        for m in METHODS:
            if m in r.index:
                rank_lists[m].append(r[m])

    rows = []
    for m in METHODS:
        arr = np.array(rank_lists[m], dtype=float)
        rows.append((m, arr.mean(), arr.std(ddof=1), len(arr)))
    rows.sort(key=lambda x: x[1])

    print(f"{'Method':8s} {'mean':>6s} {'std':>6s} {'n':>4s}")
    print('-' * 28)
    for m, mu, sd, n in rows:
        print(f"{m:8s} {mu:6.2f} {sd:6.2f} {n:4d}")


if __name__ == '__main__':
    main()
