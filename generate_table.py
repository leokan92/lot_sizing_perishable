"""Temporary script to generate the formatted longtable body from summary_metrics_table.tex"""
import re
from collections import OrderedDict

# Read new table
with open(r'src\analysis\analysis_run_20260402_175012\summary_metrics_table.tex', 'r') as f:
    lines = f.readlines()

# Parse data
data = []
for line in lines:
    line_stripped = line.strip()
    if '&' not in line_stripped:
        continue
    if any(kw in line_stripped for kw in ['toprule', 'midrule', 'bottomrule', 'Setting', 'Method']):
        continue
    
    # Remove trailing \\
    line_stripped = re.sub(r'\s*\\\\$', '', line_stripped)
    
    parts = [p.strip() for p in line_stripped.split('&')]
    if len(parts) < 11:
        continue
    
    def parse_val(s):
        s = s.strip()
        bold = '\\textbf{' in s
        s = s.replace('\\textbf{', '').replace('}', '').strip()
        try:
            return float(s), bold
        except:
            return s, bold
    
    setting = parts[0].strip()
    method = parts[1].strip()
    total_reward, total_bold = parse_val(parts[2])
    train_time, _ = parse_val(parts[4])
    eval_time, _ = parse_val(parts[5])
    wastage, wastage_bold = parse_val(parts[6])
    lost_sales, ls_bold = parse_val(parts[7])
    holding, holding_bold = parse_val(parts[8])
    inv_level, inv_bold = parse_val(parts[9])
    
    data.append({
        'setting': setting,
        'method': method,
        'total_reward': total_reward,
        'total_bold': total_bold,
        'train_time': train_time,
        'eval_time': eval_time,
        'wastage': wastage,
        'wastage_bold': wastage_bold,
        'lost_sales': lost_sales,
        'ls_bold': ls_bold,
        'holding': holding,
        'holding_bold': holding_bold,
        'inv_level': inv_level,
        'inv_bold': inv_bold,
    })

# Group by setting
settings = OrderedDict()
for row in data:
    s = row['setting']
    if s not in settings:
        settings[s] = []
    settings[s].append(row)

# Compute % vs BSP-EW
for setting, rows in settings.items():
    bspew_reward = None
    for row in rows:
        if row['method'] == 'BSP-EW':
            bspew_reward = row['total_reward']
            break
    for row in rows:
        if bspew_reward and row['method'] == 'BSP-EW':
            row['cost_red'] = 0.0
        elif bspew_reward:
            row['cost_red'] = ((row['total_reward'] - bspew_reward) / abs(bspew_reward)) * 100
        else:
            row['cost_red'] = 0.0

# Method order
method_order = ['BSP', 'BSP-EW', 'COP', 'DQN', 'EGA', 'GA', 'PPO', 'PSO', 'SAC']

def fmt_row(row, is_first, setting_name, num_methods):
    method = row['method']
    
    if row['total_bold']:
        method_str = f"\\textbf{{{method}}}"
    else:
        method_str = method
    
    if row['total_bold']:
        total_str = f"\\textbf{{{row['total_reward']:.2f}}}"
    else:
        total_str = f"{row['total_reward']:.2f}"
    
    cost_red_str = f"{row['cost_red']:.2f}"
    
    time_str = f"{row['train_time']:.2f} / {row['eval_time']:.2f}"
    
    def fmt_cost(val, bold):
        if bold:
            return f"\\textbf{{{val:.2f}}}"
        return f"{val:.2f}"
    
    w_str = fmt_cost(row['wastage'], row['wastage_bold'])
    ls_str = fmt_cost(row['lost_sales'], row['ls_bold'])
    h_str = fmt_cost(row['holding'], row['holding_bold'])
    costs_str = f"{w_str} / {ls_str} / {h_str}"
    
    if row['inv_bold']:
        inv_str = f"\\textbf{{{row['inv_level']:.2f}}}"
    else:
        inv_str = f"{row['inv_level']:.2f}"
    
    if is_first:
        prefix = f"\\multirow{{9}}{{*}}{{\\texttt{{{setting_name}}}}}"
    else:
        prefix = ""
    
    line = f"{prefix} & {method_str} & {total_str} & {cost_red_str} & {time_str} & {costs_str} & {inv_str} \\\\"
    return line

# Separate small and large
small_settings = [s for s in settings if s.startswith('I-4')]
large_settings = [s for s in settings if not s.startswith('I-4')]

output_lines = []

# Small scale
output_lines.append("% Group 1: Small-Scale Scenarios")
output_lines.append("\\multicolumn{7}{l}{\\textit{\\textbf{Group 1: Small-Scale Scenarios (4 Items, 2 Suppliers)}}} \\\\")
output_lines.append("\\midrule")

for idx, setting in enumerate(small_settings):
    rows = settings[setting]
    rows_sorted = sorted(rows, key=lambda r: method_order.index(r['method']) if r['method'] in method_order else 99)
    for i, row in enumerate(rows_sorted):
        output_lines.append(fmt_row(row, i == 0, setting, len(rows_sorted)))
    output_lines.append("\\midrule")

# Large scale
output_lines.append("")
output_lines.append("% Group 2: Large-Scale Scenarios")
output_lines.append("\\multicolumn{7}{l}{\\textit{\\textbf{Group 2: Large-Scale Scenarios (10+ Items, 4+ Suppliers)}}} \\\\")
output_lines.append("\\midrule")

for idx, setting in enumerate(large_settings):
    rows = settings[setting]
    rows_sorted = sorted(rows, key=lambda r: method_order.index(r['method']) if r['method'] in method_order else 99)
    for i, row in enumerate(rows_sorted):
        output_lines.append(fmt_row(row, i == 0, setting, len(rows_sorted)))
    output_lines.append("\\midrule")

print("=" * 80)
print("TABLE BODY:")
print("=" * 80)
for line in output_lines:
    print(line)

# Compute rankings
print("\n" + "=" * 80)
print("RANKINGS:")
print("=" * 80)

method_ranks = {m: [] for m in method_order}
for setting, rows in settings.items():
    # Sort by total_reward descending (less negative = better)
    sorted_rows = sorted(rows, key=lambda r: r['total_reward'], reverse=True)
    
    # Handle ties
    rank = 1
    i = 0
    while i < len(sorted_rows):
        # Find all tied at this value
        tied = [sorted_rows[i]]
        j = i + 1
        while j < len(sorted_rows) and abs(sorted_rows[j]['total_reward'] - sorted_rows[i]['total_reward']) < 0.01:
            tied.append(sorted_rows[j])
            j += 1
        avg_rank = sum(range(rank, rank + len(tied))) / len(tied)
        for t in tied:
            method_ranks[t['method']].append(avg_rank)
        rank += len(tied)
        i = j

print("\nAverage ranks:")
avg_ranks = {}
for method in method_order:
    ranks = method_ranks[method]
    avg = sum(ranks) / len(ranks) if ranks else 0
    avg_ranks[method] = avg
    print(f"  {method}: {avg:.2f} (ranks: {[f'{r:.1f}' for r in ranks]})")

# Sort by avg rank
sorted_methods = sorted(avg_ranks.items(), key=lambda x: x[1])
print("\nSorted by avg rank:")
for method, avg in sorted_methods:
    print(f"  {method}: {avg:.2f}")

# Wins count
print("\nWins (lowest total cost per scenario):")
wins = {m: 0 for m in method_order}
win_details = {m: [] for m in method_order}
for setting, rows in settings.items():
    best = max(rows, key=lambda r: r['total_reward'])
    # Check for ties
    best_val = best['total_reward']
    winners = [r for r in rows if abs(r['total_reward'] - best_val) < 0.01]
    for w in winners:
        if len(winners) == 1:
            wins[w['method']] += 1
        else:
            wins[w['method']] += 1.0 / len(winners)
        win_details[w['method']].append(setting)

for method in method_order:
    if wins[method] > 0:
        print(f"  {method}: {wins[method]:.1f} ({', '.join(win_details[method])})")

# Compute cost reductions for GA in large scenarios
print("\n" + "=" * 80)
print("COST REDUCTIONS FOR DISCUSSION:")
print("=" * 80)
for setting in large_settings:
    rows = settings[setting]
    bspew = next((r for r in rows if r['method'] == 'BSP-EW'), None)
    best = max(rows, key=lambda r: r['total_reward'])
    if bspew and best:
        red = ((best['total_reward'] - bspew['total_reward']) / abs(bspew['total_reward'])) * 100
        print(f"  {setting}: Best={best['method']} ({best['total_reward']:.2f}), BSP-EW={bspew['total_reward']:.2f}, Red={red:.2f}%")

# Also compute for small scenarios
print("\nSmall scenario winners:")
for setting in small_settings:
    rows = settings[setting]
    bspew = next((r for r in rows if r['method'] == 'BSP-EW'), None)
    best = max(rows, key=lambda r: r['total_reward'])
    if bspew and best:
        red = ((best['total_reward'] - bspew['total_reward']) / abs(bspew['total_reward'])) * 100
        print(f"  {setting}: Best={best['method']} ({best['total_reward']:.2f}), BSP-EW={bspew['total_reward']:.2f}, Red={red:.2f}%")

# Per-scenario winner details
print("\n" + "=" * 80)
print("PER-SCENARIO WINNERS:")
print("=" * 80)
for setting in list(small_settings) + list(large_settings):
    rows = settings[setting]
    sorted_rows = sorted(rows, key=lambda r: r['total_reward'], reverse=True)
    best = sorted_rows[0]
    bold_methods = [r['method'] for r in rows if r['total_bold']]
    print(f"  {setting}: Winner={best['method']} ({best['total_reward']:.2f}), Bold group: {bold_methods}")
