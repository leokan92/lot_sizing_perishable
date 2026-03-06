# src/analysis/results_analyzer.py

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
import os
import glob
import re # For parsing filenames
import sys # For redirecting stdout
from datetime import datetime # For timestamped directory

# --- Helper Class for Teeing stdout ---
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

# --- Configuration ---
try:
    SCRIPT_FILE_PATH = os.path.abspath(__file__)
    ANALYSIS_DIR = os.path.dirname(SCRIPT_FILE_PATH)
    found_src_dir = None
    current_path_for_src_search = ANALYSIS_DIR
    for _ in range(4):
        dir_name = os.path.basename(current_path_for_src_search)
        parent_dir_name = os.path.basename(os.path.dirname(current_path_for_src_search))
        if dir_name == 'src': found_src_dir = current_path_for_src_search; break
        if parent_dir_name == 'src': found_src_dir = os.path.dirname(current_path_for_src_search); break
        current_path_for_src_search = os.path.dirname(current_path_for_src_search)
    if found_src_dir: LOG_DIR = os.path.join(found_src_dir, 'results', 'simulation_logs')
    else:
        print("WARNING: Auto LOG_DIR failed. Using fallback."); SCRIPT_PARENT_DIR = os.path.dirname(ANALYSIS_DIR)
        LOG_DIR = os.path.join(SCRIPT_PARENT_DIR, 'results', 'simulation_logs')
except NameError:
    print("WARNING: __file__ not defined. Using CWD-relative LOG_DIR."); LOG_DIR = './src/results/simulation_logs/'

# MODIFIED: Define metrics for the summary table, including execution times
SUMMARY_METRICS_CONFIG = {
    'avg_reward': 'Avg. Total Reward', # Renamed from Total_Episode_Reward to match summary file
    'pct_vs_bspew': '% vs BSP-EW',     # Derived: (method - bspew) / bspew
    'init_train_time_s': 'Init/Train Time (s)',
    'evaluation_time_s': 'Eval Time (s)',
    'Wastage_Cost': 'Avg. Wastage Cost',
    'Lost_Sales_Cost': 'Avg. Lost Sales Cost',
    'Holding_Cost': 'Avg. Holding Cost',
    'Avg_InvLevel_All_Items': 'Avg. Inv Level (All Items)',
    'Step_Reward': 'Avg. Step Reward'
}
# MODIFIED: Order of columns in the LaTeX and text summary tables
SUMMARY_TABLE_COLUMN_ORDER = [
    'Setting',
    'Method',
    SUMMARY_METRICS_CONFIG['avg_reward'],
    SUMMARY_METRICS_CONFIG['pct_vs_bspew'],
    SUMMARY_METRICS_CONFIG['init_train_time_s'],
    SUMMARY_METRICS_CONFIG['evaluation_time_s'],
    SUMMARY_METRICS_CONFIG['Wastage_Cost'],
    SUMMARY_METRICS_CONFIG['Lost_Sales_Cost'],
    SUMMARY_METRICS_CONFIG['Holding_Cost'],
    SUMMARY_METRICS_CONFIG['Avg_InvLevel_All_Items'],
    SUMMARY_METRICS_CONFIG['Step_Reward']
]


# --- Instance Name Mapping ---
# Maps internal setting filenames to publication-ready instance names.
SETTING_TO_INSTANCE_NAME = {
    'setting_0':  'I-4-2-30-Flat',
    'setting_1':  'I-4-2-30-L',
    'setting_2':  'I-4-2-30-H-Demand',
    'setting_3':  'I-4-2-30-H-Supply',
    'setting_4':  'I-4-2-30-H-Combined',
    'setting_5':  'I-4-2-30-L-LongLT',
    'setting_6':  'I-4-2-30-L-MildSeas',
    'setting_7':  'I-4-2-30-L-RndSeas-A',
    'setting_8':  'I-4-2-30-L-RndSeas-B',
    'setting_9':  'I-4-2-90-L',
    'setting_10': 'I-10-4-30-L',
    'setting_11': 'I-10-4-30-H-Demand',
    'setting_12': 'I-10-4-30-H-Supply',
    'setting_13': 'I-10-4-30-H-Combined',
    'setting_14': 'I-10-4-60-H',
    'setting_15': 'I-20-5-60-H',
}

# Maps the raw "agent_name (agent_type)" string to a short display name.
METHOD_DISPLAY_NAME = {
    'bsp_default (bsp)':                          'BSP',
    'bsp_ew_default (bsp_ew)':                    'BSP-EW',
    'cop_default (cop)':                           'COP',
    'ga_config (pymoo_meta_heuristic)':            'GA',
    'nsga2_config (pymoo_meta_heuristic)':         'EGA',
    'pso_config (pymoo_meta_heuristic)':           'PSO',
    'stable_baselines_default (stable_baselines)': 'PPO',
    'stable_baselines_dqn (stable_baselines)':     'DQN',
    'stable_baselines_sac (stable_baselines)':     'SAC',
}

# --- Helper Functions ---
def load_latest_summary_file(log_dir):
    """Finds and loads the most recent experiment_summary_...csv file."""
    summary_files = glob.glob(os.path.join(log_dir, 'experiment_summary_*.csv'))
    if not summary_files:
        print("FATAL: No 'experiment_summary_*.csv' file found. Cannot proceed with analysis.")
        return None
    latest_file = max(summary_files, key=os.path.getctime)
    print(f"Loading summary data from: {os.path.basename(latest_file)}")
    try:
        df = pd.read_csv(latest_file)
        # Prepare for merge by creating columns that match the analyzer's conventions
        df.rename(columns={'env_name': 'Setting'}, inplace=True)
        df['Method'] = df['agent_name'] + ' (' + df['agent_type'] + ')'
        # Replace method names with short display names
        df['Method'] = df['Method'].map(METHOD_DISPLAY_NAME).fillna(df['Method'])
        # Keep original setting code for file lookups, then rename for display
        df['_setting_code'] = df['Setting']
        df['Setting'] = df['Setting'].map(SETTING_TO_INSTANCE_NAME).fillna(df['Setting'])
        return df
    except Exception as e:
        print(f"Error loading summary file {latest_file}: {e}")
        return None

def load_log_file(filepath):
    try: return pd.read_csv(filepath)
    except Exception as e: print(f"Error loading {filepath}: {e}"); return None

def get_item_inv_cols(df):
    return [col for col in df.columns if col.endswith('_InvLevel') and col.startswith('Item')]

def escape_latex(text):
    if not isinstance(text, str): return text
    return text.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

# --- Main Analysis Functions ---

def calculate_and_merge_detailed_metrics(summary_df):
    """
    Iterates through a summary dataframe, finds corresponding detail files,
    calculates metrics, and returns a new dataframe with all performance data
    (both per-step and per-episode).
    """
    all_perf_data = []
    print("\n--- Processing detailed simulation logs (*_sim_details.csv) ---")
    for idx, run_summary in summary_df.iterrows():
        # Construct the exact filename for the detailed simulation log
        # Use the original setting code (e.g. setting_1) since that is what the files are named with
        setting_code = run_summary.get('_setting_code', run_summary['Setting'])
        detail_filename = f"{setting_code}_{run_summary['agent_name']}_{run_summary['agent_type']}_seed{run_summary['seed']}_sim_details.csv"
        detail_filepath = os.path.join(LOG_DIR, detail_filename)

        detail_df = load_log_file(detail_filepath)
        if detail_df is None:
            print(f"  - WARNING: Could not find or load detail file: {detail_filename}. This run will be excluded from plots and stat tests.")
            continue

        print(f"  + Loaded details for: {run_summary['Method']} (Seed: {run_summary['seed']})")

        # Add identifiers to every row of the detail_df
        detail_df['Setting'] = run_summary['Setting']
        detail_df['Method'] = run_summary['Method']
        detail_df['seed'] = run_summary['seed']

        # Calculate Total Episode Reward and merge it back
        ep_rewards = detail_df.groupby('Episode')['Step_Reward'].sum().reset_index()
        ep_rewards = ep_rewards.rename(columns={'Step_Reward': 'Total_Episode_Reward'})
        detail_df = pd.merge(detail_df, ep_rewards, on='Episode', how='left')

        # Calculate average inventory level across items for each step
        item_inv_cols = get_item_inv_cols(detail_df)
        if item_inv_cols:
            detail_df['Avg_InvLevel_All_Items'] = detail_df[item_inv_cols].mean(axis=1)
        else:
            detail_df['Avg_InvLevel_All_Items'] = np.nan

        all_perf_data.append(detail_df)

    if not all_perf_data:
        print("ERROR: No detailed simulation data could be loaded. Cannot generate plots or statistical tests.")
        return None

    # This is a large dataframe with one row per simulation step, for all runs
    return pd.concat(all_perf_data, ignore_index=True)


def perform_visual_and_stat_analysis(setting_name, setting_perf_df, output_dir, alpha=0.05):
    """
    Generates box plots and statistical comparison tables for a given setting,
    using a pre-processed dataframe containing all necessary data.
    """
    variables_to_analyze = [
        'Total_Episode_Reward', 'Wastage_Cost', 'Lost_Sales_Cost',
        'Holding_Cost', 'Avg_InvLevel_All_Items', 'Step_Reward'
    ]

    unique_methods = setting_perf_df['Method'].unique()

    print(f"\n--- Generating Statistical Comparison Tables for {setting_name} ---")
    higher_is_better_map = {'Total_Episode_Reward': True, 'Step_Reward': True, 'Wastage_Cost': False, 'Lost_Sales_Cost': False, 'Holding_Cost': False, 'Avg_InvLevel_All_Items': False}

    for var in variables_to_analyze:
        if var not in setting_perf_df.columns: continue

        comparison_matrix = pd.DataFrame(index=unique_methods, columns=unique_methods, dtype=str)
        np.fill_diagonal(comparison_matrix.values, '-')

        data_for_var_by_method = {}
        if var == 'Total_Episode_Reward':
            unique_ep_rewards = setting_perf_df[['Method', 'seed', 'Episode', 'Total_Episode_Reward']].drop_duplicates()
            for method in unique_methods: data_for_var_by_method[method] = unique_ep_rewards[unique_ep_rewards['Method'] == method][var]
        else:
            for method in unique_methods: data_for_var_by_method[method] = setting_perf_df[setting_perf_df['Method'] == method][var].dropna()

        for i in range(len(unique_methods)):
            for j in range(i + 1, len(unique_methods)):
                method_A, method_B = unique_methods[i], unique_methods[j]
                data_A, data_B = data_for_var_by_method[method_A], data_for_var_by_method[method_B]
                if data_A.empty or data_B.empty or data_A.nunique() < 2 or data_B.nunique() < 2:
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = "NA (data)"
                    continue
                try:
                    _, p_value = mannwhitneyu(data_A, data_B, alternative='two-sided')
                    is_significant = p_value < alpha
                    result_str = "NS"
                    if is_significant:
                        median_A, median_B = data_A.median(), data_B.median()
                        is_A_better = (median_A > median_B) if higher_is_better_map.get(var, False) else (median_A < median_B)
                        result_str = f"{escape_latex(method_A) if is_A_better else escape_latex(method_B)} **"
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = result_str
                except ValueError as e:
                    print(f"    Stat test error for {var} ({method_A} vs {method_B}): {e}")
                    comparison_matrix.loc[method_A, method_B] = "Error"
        print(f"\n  Statistical Comparison Matrix for: {var} (Setting: {escape_latex(setting_name)})")
        print(f"  (Higher value is better for '{var}': {higher_is_better_map.get(var, 'N/A')})")
        print(comparison_matrix.to_string())
        matrix_filename = os.path.join(output_dir, f'{setting_name}_stat_matrix_{var.replace(" ", "_")}.csv')
        comparison_matrix.to_csv(matrix_filename)
        print(f"  Saved stat matrix: {os.path.basename(matrix_filename)}")


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank bolding logic
# ---------------------------------------------------------------------------
# For each (setting, metric) we:
#   1. Rank the methods by their mean (best first).
#   2. Compare the best against each subsequent method using a paired
#      Wilcoxon signed-rank test on per-episode values.
#   3. All methods whose distribution is NOT significantly different from
#      the best are marked as bold (they form a "statistical tie group").
#
# The function returns a dict:  bold_mask[(setting, method, display_col)] = True
# ---------------------------------------------------------------------------

# Which display-column names correspond to "higher is better"?
_HIGHER_IS_BETTER_DISPLAY = {
    SUMMARY_METRICS_CONFIG['avg_reward'],       # Avg. Total Reward
    SUMMARY_METRICS_CONFIG['Step_Reward'],       # Avg. Step Reward
}

# Metrics that should NOT receive bold treatment (execution times, derived cols, etc.)
_SKIP_BOLD_COLS = {
    SUMMARY_METRICS_CONFIG['init_train_time_s'],
    SUMMARY_METRICS_CONFIG['evaluation_time_s'],
    SUMMARY_METRICS_CONFIG['pct_vs_bspew'],
}

def _episode_level_series(performance_df, setting, method, metric_internal):
    """Return a Series indexed by Episode with one value per episode.

    For 'Total_Episode_Reward' the value is already per-episode (sum of
    step rewards).  For per-step metrics we take the *mean* across steps
    within each episode so we get one observation per episode.
    """
    mask = (performance_df['Setting'] == setting) & (performance_df['Method'] == method)
    sub = performance_df.loc[mask]
    if sub.empty or metric_internal not in sub.columns:
        return pd.Series(dtype=float)

    if metric_internal == 'Total_Episode_Reward':
        return (
            sub[['Episode', 'Total_Episode_Reward']]
            .drop_duplicates()
            .set_index('Episode')['Total_Episode_Reward']
            .sort_index()
        )
    else:
        return sub.groupby('Episode')[metric_internal].mean().sort_index()


def compute_bold_mask(performance_df, summary_df, alpha=0.05):
    """Compute which (setting, method, display_col) cells should be bolded.

    Returns
    -------
    bold_mask : dict
        Keys are (setting, method, display_col_name) tuples; values are True.
    """
    bold_mask = {}
    if performance_df is None:
        return bold_mask

    # Map from display column name -> internal column name used in performance_df
    display_to_internal = {
        SUMMARY_METRICS_CONFIG['avg_reward']: 'Total_Episode_Reward',
        SUMMARY_METRICS_CONFIG['Wastage_Cost']: 'Wastage_Cost',
        SUMMARY_METRICS_CONFIG['Lost_Sales_Cost']: 'Lost_Sales_Cost',
        SUMMARY_METRICS_CONFIG['Holding_Cost']: 'Holding_Cost',
        SUMMARY_METRICS_CONFIG['Avg_InvLevel_All_Items']: 'Avg_InvLevel_All_Items',
        SUMMARY_METRICS_CONFIG['Step_Reward']: 'Step_Reward',
    }

    settings = performance_df['Setting'].unique()
    for setting in settings:
        methods_in_setting = performance_df.loc[
            performance_df['Setting'] == setting, 'Method'
        ].unique()

        for display_col, internal_col in display_to_internal.items():
            if display_col in _SKIP_BOLD_COLS:
                continue

            higher_better = display_col in _HIGHER_IS_BETTER_DISPLAY

            # Collect per-episode distributions for every method
            method_series = {}
            method_means = {}
            for m in methods_in_setting:
                s = _episode_level_series(performance_df, setting, m, internal_col)
                if s.empty:
                    continue
                method_series[m] = s
                method_means[m] = s.mean()

            if not method_means:
                continue

            # Sort methods: best first
            sorted_methods = sorted(
                method_means.keys(),
                key=lambda m: method_means[m],
                reverse=higher_better,
            )

            best_method = sorted_methods[0]
            best_data = method_series[best_method]

            # The best is always bolded
            bold_mask[(setting, best_method, display_col)] = True

            # Compare best against each subsequent method
            for other in sorted_methods[1:]:
                other_data = method_series[other]

                # Align by episode index so the test is truly paired
                common_idx = best_data.index.intersection(other_data.index)
                if len(common_idx) < 6:
                    # Too few paired observations for a meaningful test;
                    # conservatively do NOT bold the other method.
                    break

                a = best_data.loc[common_idx].values
                b = other_data.loc[common_idx].values
                diff = a - b

                # If all differences are zero the distributions are identical
                if np.all(diff == 0):
                    bold_mask[(setting, other, display_col)] = True
                    continue

                try:
                    _, p = wilcoxon(a, b, alternative='two-sided')
                except ValueError:
                    # e.g. all-zero differences handled above, but just in case
                    bold_mask[(setting, other, display_col)] = True
                    continue

                if p >= alpha:
                    # NOT significantly different from best -> bold it too
                    bold_mask[(setting, other, display_col)] = True
                else:
                    # Significantly worse -> stop expanding the tie group
                    break

    n_bold = len(bold_mask)
    print(f"\n--- Wilcoxon Signed-Rank Bold Mask ---")
    print(f"  Total bold cells: {n_bold}  (alpha={alpha})")
    return bold_mask


def generate_summary_tables(summary_data_list, column_order, text_path, latex_path,
                            bold_mask=None):
    if not summary_data_list:
        print("No summary data to write to files.")
        return
    if bold_mask is None:
        bold_mask = {}

    df_summary = pd.DataFrame(summary_data_list)
    # Ensure all columns exist, fill with NaN if not
    for col in column_order:
        if col not in df_summary.columns: df_summary[col] = np.nan
    df_summary = df_summary[column_order]

    # Sort rows by the original setting order (as defined in SETTING_TO_INSTANCE_NAME)
    _instance_order = {name: idx for idx, name in enumerate(SETTING_TO_INSTANCE_NAME.values())}
    df_summary['_sort_key'] = df_summary['Setting'].map(_instance_order).fillna(999)
    df_summary = df_summary.sort_values(['_sort_key', 'Method']).drop(columns='_sort_key').reset_index(drop=True)

    # --- CSV/Text File ---
    df_summary_to_csv = df_summary.copy()
    for col in df_summary_to_csv.columns:
        if pd.api.types.is_numeric_dtype(df_summary_to_csv[col]):
            df_summary_to_csv[col] = df_summary_to_csv[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    df_summary_to_csv.to_csv(text_path, index=False)
    print(f"\nSummary table saved to text (CSV) file: {os.path.basename(text_path)}")

    # --- Console Output ---
    df_display = df_summary.copy()
    df_display['Setting'] = df_display['Setting'].apply(escape_latex)
    df_display['Method'] = df_display['Method'].apply(escape_latex)
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    print("\n--- Summary Table (for console log) ---")
    print(df_display.to_string(index=False))

    # --- LaTeX File (with bold best-group cells) ---
    num_metric_cols = len(column_order) - 2
    col_spec = "ll" + "r" * num_metric_cols
    latex_string = "\\begin{table}[htbp]\n  \\centering\n"
    latex_string += "  \\caption{Summary of Average Performance Metrics}\n  \\label{tab:summary_metrics_auto}\n"
    latex_string += f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
    header = " & ".join([escape_latex(col) for col in df_display.columns]) + " \\\\\n"
    latex_string += f"    {header}    \\midrule\n"

    for row_idx, row in df_summary.iterrows():
        setting_raw = row['Setting']
        method_raw = row['Method']
        cells = []
        for col in column_order:
            if col == 'Setting':
                cells.append(escape_latex(str(setting_raw)))
            elif col == 'Method':
                cells.append(escape_latex(str(method_raw)))
            else:
                val = row[col]
                formatted = f"{val:.2f}" if pd.notnull(val) else "-"
                if bold_mask.get((setting_raw, method_raw, col), False):
                    formatted = "\\textbf{" + formatted + "}"
                cells.append(formatted)
        latex_string += "    " + " & ".join(cells) + " \\\\\n"

    latex_string += "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    latex_string += "% Note: Ensure the 'booktabs' package is included in your LaTeX preamble.\n"
    latex_string += "% Bold values indicate methods belonging to the best statistical group\n"
    latex_string += "% (Wilcoxon signed-rank test, p >= 0.05 vs. the best method).\n"
    with open(latex_path, 'w', encoding='utf-8') as f: f.write(latex_string)
    print(f"\nSummary table saved to LaTeX file: {os.path.basename(latex_path)}")
    print("\n--- LaTeX Table Code ---")
    print(latex_string)


# --- Main execution ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = os.path.join(script_dir, f"analysis_run_{timestamp}")
    os.makedirs(current_run_output_dir, exist_ok=True)
    results_txt_main_log_path = os.path.join(current_run_output_dir, "analysis_full_log.txt")
    summary_text_table_path = os.path.join(current_run_output_dir, "summary_metrics_table.csv")
    summary_latex_table_path = os.path.join(current_run_output_dir, "summary_metrics_table.tex")
    original_stdout = sys.stdout
    try:
        with open(results_txt_main_log_path, 'w', encoding='utf-8') as f_out:
            sys.stdout = Tee(original_stdout, f_out)
            print(f"Analysis Run: {timestamp}\nLOG_DIR: {os.path.abspath(LOG_DIR)}\nOutput DIR: {os.path.abspath(current_run_output_dir)}\n" + "-"*50)

            # Step 1: Load the main summary file. This is our master list of experiments.
            summary_df = load_latest_summary_file(LOG_DIR)
            if summary_df is None:
                sys.exit(1) # Exit if no summary file is found

            # Step 2: Load all detailed data and create a single performance dataframe
            performance_df = calculate_and_merge_detailed_metrics(summary_df)

            # Step 3: Generate plots and statistical tests for each setting
            if performance_df is not None:
                settings_to_analyze = performance_df['Setting'].unique()
                print(f"\n--- Starting Visual and Statistical Analysis for settings: {list(settings_to_analyze)} ---")
                for setting in settings_to_analyze:
                    print(f"\n{'-'*20} Analyzing Setting: {setting} {'-'*20}")
                    setting_perf_df = performance_df[performance_df['Setting'] == setting]
                    perform_visual_and_stat_analysis(setting, setting_perf_df, current_run_output_dir)
            else:
                settings_to_analyze = []

            # Step 4: Create the final summary table data
            # We average the results for each method across all its seeds.
            final_summary_list = []
            # Base the final summary on the original summary_df to include runs even if detail files were missing
            grouped_summary = summary_df.groupby(['Setting', 'Method'])

            # Calculate means for the initial summary data (rewards, times)
            summary_means = grouped_summary[
                ['avg_reward', 'init_train_time_s', 'evaluation_time_s']
            ].mean().reset_index()

            # If we have performance data, calculate its means too
            if performance_df is not None:
                grouped_perf = performance_df.groupby(['Setting', 'Method'])
                perf_means = grouped_perf[
                    ['Wastage_Cost', 'Lost_Sales_Cost', 'Holding_Cost', 'Avg_InvLevel_All_Items', 'Step_Reward']
                ].mean().reset_index()
                # Merge the performance means into the summary means
                final_summary_df = pd.merge(summary_means, perf_means, on=['Setting', 'Method'], how='left')
            else:
                final_summary_df = summary_means

            # Convert dataframe to the list of dictionaries format required by the table generator
            # And rename columns to their display names
            display_name_map = {v: k for k, v in SUMMARY_METRICS_CONFIG.items()}
            final_summary_df.rename(columns=display_name_map, inplace=True) # This is a bit tricky, let's build dicts directly
            
            summary_data_for_table = []
            for _, row in final_summary_df.iterrows():
                row_dict = row.to_dict()
                # Now rename keys to the display names
                renamed_dict = {}
                for key, value in row_dict.items():
                    renamed_dict[SUMMARY_METRICS_CONFIG.get(key, key)] = value
                summary_data_for_table.append(renamed_dict)


            # Step 5: Compute % vs BSP-EW for each setting
            pct_col = SUMMARY_METRICS_CONFIG['pct_vs_bspew']
            reward_col = SUMMARY_METRICS_CONFIG['avg_reward']
            bspew_reward_by_setting = {}
            for rd in summary_data_for_table:
                method_str = rd.get('Method', '')
                if 'bsp_ew' in method_str.lower():
                    bspew_reward_by_setting[rd['Setting']] = rd.get(reward_col, np.nan)
            for rd in summary_data_for_table:
                bspew_val = bspew_reward_by_setting.get(rd['Setting'], np.nan)
                method_val = rd.get(reward_col, np.nan)
                if pd.notnull(bspew_val) and bspew_val != 0 and pd.notnull(method_val):
                    rd[pct_col] = (method_val - bspew_val) / bspew_val
                else:
                    rd[pct_col] = np.nan

            # Step 6: Compute Wilcoxon bold mask
            bold_mask = compute_bold_mask(performance_df, summary_df, alpha=0.05)

            # Step 7: Generate and save the final summary tables
            generate_summary_tables(summary_data_for_table, SUMMARY_TABLE_COLUMN_ORDER,
                                    summary_text_table_path, summary_latex_table_path,
                                    bold_mask=bold_mask)

            print("-" * 50 + "\nAnalysis script finished successfully.")
    finally:
        # Restore stdout and print final messages
        if isinstance(sys.stdout, Tee):
            for f_tee in sys.stdout.files:
                if f_tee != original_stdout and hasattr(f_tee, 'close') and not f_tee.closed:
                    f_tee.close()
        sys.stdout = original_stdout
        print(f"\nFull analysis log saved to: {os.path.abspath(results_txt_main_log_path)}")
        if 'summary_data_for_table' in locals() and summary_data_for_table:
            print(f"Summary table (CSV) saved to: {os.path.abspath(summary_text_table_path)}")
            print(f"Summary table (LaTeX) saved to: {os.path.abspath(summary_latex_table_path)}")
        print(f"All plots and detailed stat matrices saved in: {os.path.abspath(current_run_output_dir)}")