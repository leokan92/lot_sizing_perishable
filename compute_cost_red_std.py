"""Compute, per instance, the best method's cost reduction (% vs BSP-EW) and the
standard deviation of that reduction across the paired evaluation episodes.

Cost reduction per episode e (paired by (seed, Episode)):
    red_e = (R_method_e - R_bspew_e) / |mean(R_bspew)| * 100
so that mean(red_e) == (mean R_method - mean R_bspew) / |mean R_bspew| * 100,
which equals the value already reported in the summary table.
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd

MAIN_TEX = os.path.join(os.path.dirname(__file__), 'tex_folder', 'main.tex')


def parse_table_costs(path):
    """Return {instance: displayed_total_cost} parsed from the main.tex longtable."""
    costs = {}
    row_re = re.compile(r'\\texttt\{([^}]+)\}\s*&[^&]*&\s*(-?\d+\.\d+)\s*&')
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = row_re.search(line)
            if m:
                costs[m.group(1)] = float(m.group(2))
    return costs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'analysis'))
from results_analyzer import SETTING_TO_INSTANCE_NAME, METHOD_DISPLAY_NAME  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(__file__), 'src', 'results', 'simulation_logs')
SUMMARY = glob.glob(os.path.join(LOG_DIR, 'experiment_summary_*full50*.csv'))[0]

# Instances belonging to the table, in table order (skip ones not in the table).
TABLE_INSTANCES = [
    'I-4-2-30-Flat', 'I-4-2-30-L', 'I-4-2-30-H-Demand', 'I-4-2-30-H-Supply',
    'I-4-2-30-H-Combined', 'I-4-2-30-M-Demand', 'I-4-2-30-MH-Demand', 'I-4-2-30-M-Supply',
    'I-4-2-30-L-LongLT', 'I-4-2-30-L-LT1', 'I-4-2-30-L-LT5', 'I-4-2-30-L-LT7',
    'I-4-2-30-L-MildSeas', 'I-4-2-30-L-RndSeas-A', 'I-4-2-30-L-RndSeas-B', 'I-4-2-30-L-Bimodal',
    'I-4-2-30-L-Trend', 'I-4-2-30-L-StrongSeas', 'I-4-2-30-L-SL2', 'I-4-2-30-L-SL3',
    'I-4-2-30-L-SL5', 'I-4-2-30-L-SL6', 'I-4-2-30-L-HighHold', 'I-4-2-30-L-HighShort',
    'I-4-2-30-L-HighFixed', 'I-4-2-30-L-LowDem', 'I-4-2-30-L-HighDem',
    'I-4-2-60-L', 'I-4-2-90-L', 'I-4-2-90-H-Combined',
    'I-6-3-30-L', 'I-6-3-30-H-Combined', 'I-6-3-45-L-MildSeas', 'I-8-3-30-L',
    'I-8-3-30-H-Demand', 'I-8-3-45-L-LongLT',
    'I-10-4-30-L', 'I-10-4-30-H-Demand', 'I-10-4-30-H-Supply', 'I-10-4-30-H-Combined',
    'I-10-4-30-L-LongLT', 'I-10-4-30-L-MildSeas', 'I-10-4-30-L-SL5',
    'I-10-4-60-L', 'I-10-4-60-H',
    'I-15-4-30-L', 'I-15-4-60-H-Combined',
    'I-20-5-30-L', 'I-20-5-60-H', 'I-20-5-90-H-Combined',
]

INSTANCE_TO_SETTING = {v: k for k, v in SETTING_TO_INSTANCE_NAME.items()}


def episode_rewards(setting_code, agent_name, agent_type, seed):
    """Return a Series indexed by (seed, Episode) of total episode rewards."""
    fn = f"{setting_code}_{agent_name}_{agent_type}_seed{seed}_sim_details.csv"
    path = os.path.join(LOG_DIR, fn)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=['Episode', 'Step_Reward'])
    s = df.groupby('Episode')['Step_Reward'].sum()
    s.index = pd.MultiIndex.from_product([[seed], s.index], names=['seed', 'Episode'])
    return s


def main():
    summ = pd.read_csv(SUMMARY)
    summ.rename(columns={'env_name': 'Setting'}, inplace=True)
    summ['Method'] =(summ['agent_name'] + ' (' + summ['agent_type'] + ')').map(
        METHOD_DISPLAY_NAME).fillna(summ['agent_name'])

    table_costs = parse_table_costs(MAIN_TEX)

    print(f"{'Instance':28s} {'Shown':6s} {'Reward':>10s} {'CostRed%':>9s} {'Std':>7s} {'n':>4s}")
    print('-' * 70)
    out = {}
    for inst in TABLE_INSTANCES:
        setting = INSTANCE_TO_SETTING[inst]
        rows = summ[summ['Setting'] == setting]
        if rows.empty:
            print(f"{inst:28s}  (no rows)")
            continue
        # mean reward per method (avg over seeds if several)
        by_method = rows.groupby('Method').agg(
            avg_reward=('avg_reward', 'mean')).reset_index()
        bspew = by_method[by_method['Method'] == 'BSP-EW']
        if bspew.empty:
            print(f"{inst:28s}  (no BSP-EW)")
            continue
        bspew_mean = bspew['avg_reward'].iloc[0]
        # pick the method whose reward matches the value shown in the table
        shown_cost = table_costs.get(inst)
        if shown_cost is not None:
            by_method['gap'] = (by_method['avg_reward'] - shown_cost).abs()
            best = by_method.loc[by_method['gap'].idxmin()]
        else:
            best = by_method.loc[by_method['avg_reward'].idxmax()]
        best_method = best['Method']
        best_reward = best['avg_reward']
        cost_red = (best_reward - bspew_mean) / abs(bspew_mean) * 100.0

        # paired per-episode std
        seeds = sorted(rows['seed'].unique())
        red_parts = []
        for seed in seeds:
            br = rows[(rows['Method'] == best_method) & (rows['seed'] == seed)]
            er = rows[(rows['Method'] == 'BSP-EW') & (rows['seed'] == seed)]
            if br.empty or er.empty:
                continue
            r_best = episode_rewards(setting, br['agent_name'].iloc[0],
                                     br['agent_type'].iloc[0], seed)
            r_bspew = episode_rewards(setting, er['agent_name'].iloc[0],
                                      er['agent_type'].iloc[0], seed)
            if r_best is None or r_bspew is None:
                continue
            aligned = pd.concat([r_best, r_bspew], axis=1, join='inner')
            red_parts.append((aligned.iloc[:, 0] - aligned.iloc[:, 1]))
        if red_parts:
            diff = pd.concat(red_parts)
            red_e = diff / abs(bspew_mean) * 100.0
            std = red_e.std(ddof=1)
            n = len(red_e)
            mean_check = red_e.mean()
        else:
            std, n, mean_check = float('nan'), 0, float('nan')

        out[inst] = (best_method, best_reward, cost_red, std, n, mean_check)
        print(f"{inst:28s} {best_method:6s} {best_reward:10.2f} "
              f"{cost_red:9.2f} {std:7.2f} {n:4d}")

    return out


def rewrite_table(stds):
    """Rewrite the summary longtable in main.tex: drop Notes column, render
    Cost Red. as mean +/- s.d."""
    with open(MAIN_TEX, encoding='utf-8') as f:
        lines = f.readlines()

    start = next(i for i, l in enumerate(lines)
                 if l.startswith(r'\begin{longtable}{@{}l l S[table-format=-6.2]'))
    end = next(i for i in range(start, len(lines))
               if lines[i].startswith(r'\end{longtable}'))

    row_re = re.compile(r'^\\texttt\{([^}]+)\}\s*&(.*?)&(.*?)&(.*?)&(.*?)\\\\\s*$')
    new = []
    for l in lines[start:end + 1]:
        s = l.rstrip('\n')
        if s.startswith(r'\begin{longtable}'):
            new.append(r'\begin{longtable}{@{}l l S[table-format=-6.2] c@{}}' + '\n')
            continue
        if r'\caption{' in s:
            s = s.replace(
                'relative to the BSP-EW method.',
                'relative to the BSP-EW method, reported as mean $\\pm$ '
                'standard deviation across the 50 paired evaluation episodes.')
            new.append(s + '\n')
            continue
        if r'\multicolumn{5}' in s:
            new.append(s.replace(r'\multicolumn{5}', r'\multicolumn{4}') + '\n')
            continue
        if r'\textbf{Instance}' in s:
            new.append(s.replace(r' & \textbf{Notes}', '') + '\n')
            continue
        m = row_re.match(s)
        if m:
            inst, methods, cost, red, _notes = m.groups()
            std = stds[inst][3]
            new.append(f"\\texttt{{{inst}}} &{methods}&{cost}& "
                       f"${red.strip()} \\pm {std:.2f}$ \\\\\n")
            continue
        new.append(l)

    lines[start:end + 1] = new
    with open(MAIN_TEX, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nRewrote longtable in {MAIN_TEX} (lines {start+1}-{end+1}).")


if __name__ == '__main__':
    out = main()
    if '--write' in sys.argv:
        rewrite_table(out)
