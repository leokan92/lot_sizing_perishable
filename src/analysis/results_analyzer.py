# src/analysis/results_analyzer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu
import os
import glob
import re # For parsing filenames
import sys # For redirecting stdout
from datetime import datetime # For timestamped directory
from collections import defaultdict

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
    'Total_Episode_Reward': 'Avg. Total Reward',
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
    SUMMARY_METRICS_CONFIG['Total_Episode_Reward'],
    SUMMARY_METRICS_CONFIG['init_train_time_s'],
    SUMMARY_METRICS_CONFIG['evaluation_time_s'],
    SUMMARY_METRICS_CONFIG['Wastage_Cost'],
    SUMMARY_METRICS_CONFIG['Lost_Sales_Cost'],
    SUMMARY_METRICS_CONFIG['Holding_Cost'],
    SUMMARY_METRICS_CONFIG['Avg_InvLevel_All_Items'],
    SUMMARY_METRICS_CONFIG['Step_Reward']
]


# --- Helper Functions ---
# ADDED: New helper function to load timing data from the runner's summary file
def load_latest_summary_file(log_dir):
    """Finds and loads the most recent experiment_summary_...csv file."""
    summary_files = glob.glob(os.path.join(log_dir, 'experiment_summary_*.csv'))
    if not summary_files:
        print("Warning: No 'experiment_summary_*.csv' file found. Timing data will not be available.")
        return None
    latest_file = max(summary_files, key=os.path.getctime)
    print(f"Loading timing data from: {os.path.basename(latest_file)}")
    try:
        df = pd.read_csv(latest_file)
        # Prepare for merge by creating columns that match the analyzer's conventions
        df.rename(columns={'env_name': 'setting'}, inplace=True)
        df['Method'] = df['agent_name'] + ' (' + df['agent_type'] + ')'
        return df
    except Exception as e:
        print(f"Error loading summary file {latest_file}: {e}")
        return None

def parse_experiment_details_from_filename(filename_path):
    basename = os.path.basename(filename_path)
    match = re.match(r'^(.*?)_([^_]+)_seed(\d+)_sim_details\.csv$', basename)
    if match:
        full_prefix_before_type_and_seed, agent_type_from_filename, seed_str = match.groups()
        seed = int(seed_str)
        setting = "default_setting"
        agent_name = full_prefix_before_type_and_seed
        setting_match = re.match(r'^(setting_\d+)(?:_(.*))?$', full_prefix_before_type_and_seed)
        if setting_match:
            setting = setting_match.group(1)
            potential_agent_name_part = setting_match.group(2)
            if potential_agent_name_part:
                agent_name = potential_agent_name_part
            else:
                agent_name = agent_type_from_filename
        else:
            agent_name = full_prefix_before_type_and_seed
            if agent_name.endswith(f"_{agent_type_from_filename}") and len(agent_name) > len(agent_type_from_filename) + 1:
                agent_name = agent_name[:-len(f"_{agent_type_from_filename}")-1]
        return {'setting': setting, 'agent_name': agent_name, 'agent_type': agent_type_from_filename, 'seed': seed, 'filepath': filename_path}
    return None

def load_log_file(filepath):
    try: return pd.read_csv(filepath)
    except Exception as e: print(f"Error loading {filepath}: {e}"); return None

def get_item_inv_cols(df):
    return [col for col in df.columns if col.endswith('_InvLevel') and col.startswith('Item')]

def escape_latex(text):
    if not isinstance(text, str): return text
    return text.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


# --- Analysis Orchestration ---
# MODIFIED: Function signature to accept timing data
def perform_setting_analysis(setting_name, variables_to_analyze, output_dir, alpha=0.05, timing_df=None):
    print(f"\n\n--- Analyzing Setting: {setting_name} ---")

    if setting_name == "default_setting":
        all_files_pattern = os.path.join(LOG_DIR, f"*_sim_details.csv")
        all_log_files = glob.glob(all_files_pattern)
        log_files = [f for f in all_log_files if parse_experiment_details_from_filename(f) and parse_experiment_details_from_filename(f)['setting'] == "default_setting"]
    else:
        setting_files_pattern = os.path.join(LOG_DIR, f"{setting_name}_*_sim_details.csv")
        log_files = glob.glob(setting_files_pattern)

    if not log_files:
        print(f"No log files found for setting '{setting_name}'.")
        return []

    all_data_frames = []
    for f_path in log_files:
        details = parse_experiment_details_from_filename(f_path)
        if not details or details.get('setting') != setting_name: continue
        df = load_log_file(f_path)
        if df is not None:
            df['Method'] = f"{details['agent_name']} ({details['agent_type']})"
            df['Seed'] = details['seed']
            df['Filepath'] = f_path
            ep_rewards = df.groupby('Episode')['Step_Reward'].sum().reset_index().rename(columns={'Step_Reward': 'Total_Episode_Reward'})
            df = pd.merge(df, ep_rewards, on='Episode', how='left')
            item_inv_cols = get_item_inv_cols(df)
            df['Avg_InvLevel_All_Items'] = df[item_inv_cols].mean(axis=1) if item_inv_cols else np.nan
            all_data_frames.append(df)

    if not all_data_frames:
        print(f"No data successfully loaded for setting '{setting_name}'.")
        return []

    combined_df = pd.concat(all_data_frames, ignore_index=True)

    print(f"\nGenerating Box Plots for {setting_name}...")
    unique_methods = combined_df['Method'].unique()
    for var in variables_to_analyze:
        if var not in combined_df.columns:
            print(f"  Variable '{var}' not found in combined data for box plot. Skipping.")
            continue
        plt.figure(figsize=(max(8, 2.5 * len(unique_methods)), 6))
        plot_title_suffix = "(Per Episode)" if var == 'Total_Episode_Reward' else "(Per Step)"
        plot_data = combined_df[['Method', 'Seed', 'Episode', 'Total_Episode_Reward']].drop_duplicates() if var == 'Total_Episode_Reward' else combined_df
        sns.boxplot(x='Method', y=var, data=plot_data)
        plt.title(f'{var} Comparison for {escape_latex(setting_name)} {plot_title_suffix}')
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'{setting_name}_boxplot_{var.replace(" ", "_")}.png')
        plt.savefig(plot_filename); plt.close()
        print(f"  Saved box plot: {os.path.basename(plot_filename)}")

    print(f"\nGenerating Statistical Comparison Tables for {setting_name}...")
    higher_is_better_map = {'Total_Episode_Reward': True, 'Step_Reward': True, 'Wastage_Cost': False, 'Lost_Sales_Cost': False, 'Holding_Cost': False, 'Avg_InvLevel_All_Items': False}
    for item_col in get_item_inv_cols(combined_df):
        if item_col not in higher_is_better_map: higher_is_better_map[item_col] = False
    for var in variables_to_analyze:
        if var not in combined_df.columns: continue
        methods = combined_df['Method'].unique()
        comparison_matrix = pd.DataFrame(index=methods, columns=methods, dtype=str)
        np.fill_diagonal(comparison_matrix.values, '-')
        data_for_var_by_method = {}
        if var == 'Total_Episode_Reward':
            unique_ep_rewards = combined_df[['Method', 'Seed', 'Episode', 'Total_Episode_Reward']].drop_duplicates()
            for method in methods: data_for_var_by_method[method] = unique_ep_rewards[unique_ep_rewards['Method'] == method][var]
        else:
            for method in methods: data_for_var_by_method[method] = combined_df[combined_df['Method'] == method][var].dropna()
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method_A, method_B = methods[i], methods[j]
                data_A, data_B = data_for_var_by_method[method_A], data_for_var_by_method[method_B]
                if data_A.empty or data_B.empty or data_A.nunique() < 2 or data_B.nunique() < 2:
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = "NA (data)"
                    continue
                try:
                    stat, p_value = mannwhitneyu(data_A, data_B, alternative='two-sided')
                    is_significant = p_value < alpha
                    result_str = "NS"
                    if is_significant:
                        median_A, median_B = data_A.median(), data_B.median()
                        is_A_better = (median_A > median_B) if higher_is_better_map.get(var, False) else (median_A < median_B)
                        result_str = f"{escape_latex(method_A) if is_A_better else escape_latex(method_B)} **"
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = result_str
                except ValueError as e:
                    print(f"    Stat test error for {var} ({method_A} vs {method_B}): {e}")
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = "Error"
        print(f"\n  Statistical Comparison Matrix for: {var} (Setting: {escape_latex(setting_name)})")
        print(f"  (Higher value is better for '{var}': {higher_is_better_map.get(var, 'N/A (assuming lower)')})")
        temp_comparison_matrix_for_output = comparison_matrix.copy()
        for r_idx, r_val in enumerate(methods): temp_comparison_matrix_for_output.rename(index={r_val: escape_latex(r_val)}, inplace=True)
        for c_idx, c_val in enumerate(methods): temp_comparison_matrix_for_output.rename(columns={c_val: escape_latex(c_val)}, inplace=True)
        print(temp_comparison_matrix_for_output.to_string())
        matrix_filename = os.path.join(output_dir, f'{setting_name}_stat_matrix_{var.replace(" ", "_")}.csv')
        comparison_matrix.to_csv(matrix_filename)
        print(f"  Saved stat matrix: {os.path.basename(matrix_filename)}")

    # MODIFIED: Calculate averages for summary table, now including timing data
    setting_summary_rows = []
    for method_name_orig, group_df in combined_df.groupby('Method'):
        summary_row = {'Setting': setting_name, 'Method': method_name_orig}

        # Calculate metrics from the detailed simulation logs (_sim_details.csv)
        for original_metric_name, display_metric_name in SUMMARY_METRICS_CONFIG.items():
            if original_metric_name in ['init_train_time_s', 'evaluation_time_s']: continue # Skip time metrics here
            if original_metric_name == 'Total_Episode_Reward':
                if 'Total_Episode_Reward' in group_df.columns:
                    ep_rewards_method = group_df[['Episode', 'Seed', 'Total_Episode_Reward']].drop_duplicates()
                    summary_row[display_metric_name] = ep_rewards_method['Total_Episode_Reward'].mean()
                else: summary_row[display_metric_name] = np.nan
            elif original_metric_name in group_df.columns:
                summary_row[display_metric_name] = group_df[original_metric_name].mean()
            else: summary_row[display_metric_name] = np.nan

        # ADDED: Add timing metrics from the summary CSV (experiment_summary_... .csv)
        if timing_df is not None:
            method_timing_df = timing_df[(timing_df['setting'] == setting_name) & (timing_df['Method'] == method_name_orig)]
            if not method_timing_df.empty:
                summary_row[SUMMARY_METRICS_CONFIG['init_train_time_s']] = method_timing_df['init_train_time_s'].mean()
                summary_row[SUMMARY_METRICS_CONFIG['evaluation_time_s']] = method_timing_df['evaluation_time_s'].mean()
            else:
                summary_row[SUMMARY_METRICS_CONFIG['init_train_time_s']] = np.nan
                summary_row[SUMMARY_METRICS_CONFIG['evaluation_time_s']] = np.nan
        else: # If no timing_df was provided at all
            summary_row[SUMMARY_METRICS_CONFIG['init_train_time_s']] = np.nan
            summary_row[SUMMARY_METRICS_CONFIG['evaluation_time_s']] = np.nan

        setting_summary_rows.append(summary_row)

    return setting_summary_rows


def generate_summary_text_file(all_summary_data, column_order, filepath):
    if not all_summary_data:
        print("No summary data to write to text file.")
        return
    df_summary = pd.DataFrame(all_summary_data)
    for col in column_order:
        if col not in df_summary.columns: df_summary[col] = np.nan
    df_summary = df_summary[column_order]
    df_display_summary = df_summary.copy()
    df_display_summary['Setting'] = df_display_summary['Setting'].apply(escape_latex)
    df_display_summary['Method'] = df_display_summary['Method'].apply(escape_latex)
    for col in df_display_summary.columns:
        if pd.api.types.is_numeric_dtype(df_display_summary[col]):
            df_display_summary[col] = df_display_summary[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    df_summary_to_csv = df_summary.copy()
    for col in df_summary_to_csv.columns:
        if pd.api.types.is_numeric_dtype(df_summary_to_csv[col]):
            df_summary_to_csv[col] = df_summary_to_csv[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    df_summary_to_csv.to_csv(filepath, index=False)
    print(f"\nSummary table saved to text (CSV) file: {os.path.basename(filepath)}")
    print("\n--- Summary Table (for console log, with LaTeX escapes on text) ---")
    print(df_display_summary.to_string(index=False))


def generate_latex_summary_table(all_summary_data, column_order, filepath):
    if not all_summary_data:
        print("No summary data to write to LaTeX file.")
        return
    df_summary = pd.DataFrame(all_summary_data)
    for col in column_order:
        if col not in df_summary.columns: df_summary[col] = np.nan
    df_summary = df_summary[column_order]
    num_metric_cols = len(column_order) - 2
    col_spec = "ll" + "r" * num_metric_cols
    latex_string = "\\begin{table}[htbp]\n  \\centering\n"
    latex_string += "  \\caption{Summary of Average Performance Metrics Across Settings and Methods}\n"
    latex_string += "  \\label{tab:summary_metrics_auto}\n"
    latex_string += f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
    header = " & ".join([escape_latex(col) for col in df_summary.columns]) + " \\\\\n"
    latex_string += f"    {header}    \\midrule\n"
    for _, row in df_summary.iterrows():
        row_values = []
        for col_name in df_summary.columns:
            val = row[col_name]
            if isinstance(val, (float, np.number)):
                row_values.append(f"{val:.2f}" if pd.notnull(val) else "-")
            else:
                row_values.append(escape_latex(str(val)))
        latex_string += "    " + " & ".join(row_values) + " \\\\\n"
    latex_string += "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    latex_string += "% Note: Ensure the 'booktabs' package (\\usepackage{booktabs}) is included in your LaTeX preamble.\n"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex_string)
    print(f"\nSummary table saved to LaTeX file: {os.path.basename(filepath)}")
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

            # ADDED: Load timing data at the start of the analysis
            timing_df = load_latest_summary_file(LOG_DIR)

            all_csv_in_log = glob.glob(os.path.join(LOG_DIR, '*_sim_details.csv')) # Only look for detail files
            print(f"Diagnostic: Found {len(all_csv_in_log)} '*_sim_details.csv' files in LOG_DIR. First 3: {all_csv_in_log[:3] if all_csv_in_log else 'None'}\n" + "-"*50)

            discovered_settings = set()
            if all_csv_in_log:
                for f_path in all_csv_in_log:
                    details = parse_experiment_details_from_filename(f_path)
                    if details: discovered_settings.add(details['setting'])
            if not discovered_settings:
                print("No settings discovered from CSV filenames. Using default 'setting_1'.")
                settings_to_analyze = ["setting_1"] # Fallback
            else:
                settings_to_analyze = sorted(list(discovered_settings))
                if "default_setting" in settings_to_analyze and len(settings_to_analyze) > 1:
                    settings_to_analyze.remove("default_setting")
                    settings_to_analyze.append("default_setting")
            print(f"Analyzing settings: {settings_to_analyze}")

            variables_for_analysis = ['Total_Episode_Reward', 'Wastage_Cost', 'Lost_Sales_Cost', 'Holding_Cost', 'Avg_InvLevel_All_Items', 'Step_Reward']
            all_settings_summary_data = []
            for setting in settings_to_analyze:
                summary_data_for_setting = perform_setting_analysis(setting, variables_for_analysis, current_run_output_dir, timing_df=timing_df)
                if summary_data_for_setting:
                    all_settings_summary_data.extend(summary_data_for_setting)

            if all_settings_summary_data:
                generate_summary_text_file(all_settings_summary_data, SUMMARY_TABLE_COLUMN_ORDER, summary_text_table_path)
                generate_latex_summary_table(all_settings_summary_data, SUMMARY_TABLE_COLUMN_ORDER, summary_latex_table_path)
            else:
                print("\nNo summary data collected from any setting. Summary tables will not be generated.")
            print("-" * 50 + "\nAnalysis script finished successfully.")
    finally:
        if isinstance(sys.stdout, Tee):
            for f_tee in sys.stdout.files:
                if f_tee != original_stdout and hasattr(f_tee, 'close') and not f_tee.closed: f_tee.close()
        sys.stdout = original_stdout
        print(f"\nFull analysis log saved to: {os.path.abspath(results_txt_main_log_path)}")
        if 'all_settings_summary_data' in locals() and all_settings_summary_data:
            print(f"Summary table (CSV) saved to: {os.path.abspath(summary_text_table_path)}")
            print(f"Summary table (LaTeX) saved to: {os.path.abspath(summary_latex_table_path)}")
        print(f"All plots and detailed stat matrices saved in: {os.path.abspath(current_run_output_dir)}")