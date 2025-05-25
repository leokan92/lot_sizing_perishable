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

# Define metrics for the summary table and their desired names/order
SUMMARY_METRICS_CONFIG = {
    'Total_Episode_Reward': 'Avg. Total Reward',
    'Wastage_Cost': 'Avg. Wastage Cost',
    'Lost_Sales_Cost': 'Avg. Lost Sales Cost',
    'Holding_Cost': 'Avg. Holding Cost',
    'Avg_InvLevel_All_Items': 'Avg. Inv Level (All Items)',
    'Step_Reward': 'Avg. Step Reward'
}
# Order of columns in the LaTeX and text summary tables
SUMMARY_TABLE_COLUMN_ORDER = [
    'Setting', 
    'Method', 
    SUMMARY_METRICS_CONFIG['Total_Episode_Reward'],
    SUMMARY_METRICS_CONFIG['Wastage_Cost'],
    SUMMARY_METRICS_CONFIG['Lost_Sales_Cost'],
    SUMMARY_METRICS_CONFIG['Holding_Cost'],
    SUMMARY_METRICS_CONFIG['Avg_InvLevel_All_Items'],
    SUMMARY_METRICS_CONFIG['Step_Reward']
]


# --- Helper Functions ---
def parse_experiment_details_from_filename(filename_path):
    basename = os.path.basename(filename_path)
    # Regex to capture: (setting_X_agentName_optionalAgentType)_actualAgentType_seedY_sim_details.csv
    # Or: (agentName_optionalAgentType)_actualAgentType_seedY_sim_details.csv
    match = re.match(r'^(.*?)_([^_]+)_seed(\d+)_sim_details\.csv$', basename)
    if match:
        full_prefix_before_type_and_seed, agent_type_from_filename, seed_str = match.groups()
        seed = int(seed_str)
        
        setting = "default_setting" # Default
        agent_name = full_prefix_before_type_and_seed # Default

        # Try to parse setting_X
        setting_match = re.match(r'^(setting_\d+)(?:_(.*))?$', full_prefix_before_type_and_seed)
        if setting_match:
            setting = setting_match.group(1)
            potential_agent_name_part = setting_match.group(2)
            if potential_agent_name_part: # If there's something after setting_X_
                agent_name = potential_agent_name_part
            else: # If it's just "setting_X", agent_name might be the agent_type_from_filename
                agent_name = agent_type_from_filename 
        else: # No "setting_X" prefix, full_prefix_before_type_and_seed is likely the agent name
            agent_name = full_prefix_before_type_and_seed
            # If agent_name itself ends with the agent_type, remove it for a cleaner agent_name
            if agent_name.endswith(f"_{agent_type_from_filename}") and len(agent_name) > len(agent_type_from_filename) + 1:
                agent_name = agent_name[:-len(f"_{agent_type_from_filename}")-1]
            elif agent_name == agent_type_from_filename: # If agent_name is identical to agent_type
                pass # Keep agent_name as is, it's just the type

        return {'setting': setting, 'agent_name': agent_name, 'agent_type': agent_type_from_filename, 'seed': seed, 'filepath': filename_path}
    return None


def load_log_file(filepath):
    try: return pd.read_csv(filepath)
    except Exception as e: print(f"Error loading {filepath}: {e}"); return None

def get_item_inv_cols(df):
    return [col for col in df.columns if col.endswith('_InvLevel') and col.startswith('Item')]

def escape_latex(text):
    """Basic LaTeX escaping for strings."""
    if not isinstance(text, str):
        return text
    return text.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


# --- Analysis Orchestration ---
def perform_setting_analysis(setting_name, variables_to_analyze, output_dir, alpha=0.05):
    print(f"\n\n--- Analyzing Setting: {setting_name} ---")
    
    # Construct pattern based on whether it's a specific setting or default
    if setting_name == "default_setting":
        # Glob all, then filter
        all_files_pattern = os.path.join(LOG_DIR, f"*_sim_details.csv")
        all_log_files = glob.glob(all_files_pattern)
        # Filter out files that DO match a specific "setting_X" pattern if we're looking for "default_setting"
        log_files = [f for f in all_log_files if parse_experiment_details_from_filename(f) and parse_experiment_details_from_filename(f)['setting'] == "default_setting"]
    else:
        setting_files_pattern = os.path.join(LOG_DIR, f"{setting_name}_*_sim_details.csv")
        log_files = glob.glob(setting_files_pattern)


    if not log_files:
        print(f"No log files found for setting '{setting_name}'.")
        return [] # Return empty list if no data

    all_data_frames = []
    for f_path in log_files:
        details = parse_experiment_details_from_filename(f_path)
        
        # This check is now implicitly handled by how log_files are gathered for default_setting vs specific settings.
        # However, an explicit check for safety is not bad.
        if not details or details.get('setting') != setting_name:
            continue # Skip if details don't match the current setting analysis pass

        df = load_log_file(f_path)
        if df is not None: # details is already confirmed to exist and match setting_name
            df['Method'] = f"{details['agent_name']} ({details['agent_type']})"
            df['Seed'] = details['seed']
            df['Filepath'] = f_path 

            ep_rewards = df.groupby('Episode')['Step_Reward'].sum().reset_index()
            ep_rewards = ep_rewards.rename(columns={'Step_Reward': 'Total_Episode_Reward'})
            df = pd.merge(df, ep_rewards, on='Episode', how='left')
            
            item_inv_cols = get_item_inv_cols(df)
            if item_inv_cols:
                df['Avg_InvLevel_All_Items'] = df[item_inv_cols].mean(axis=1)
            else:
                df['Avg_InvLevel_All_Items'] = np.nan
            
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
        plot_title_suffix = ""

        if var == 'Total_Episode_Reward':
            plot_data = combined_df[['Method', 'Seed', 'Episode', 'Total_Episode_Reward']].drop_duplicates()
            sns.boxplot(x='Method', y=var, data=plot_data)
            plot_title_suffix = "(Per Episode)"
        else:
            sns.boxplot(x='Method', y=var, data=combined_df)
            plot_title_suffix = "(Per Step)"

        # Use original setting_name for plot titles, escape it for LaTeX if title goes there.
        # For filenames, use setting_name directly as it should be file-system safe.
        plt.title(f'{var} Comparison for {escape_latex(setting_name)} {plot_title_suffix}')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # CORRECTED FILENAME GENERATION: Do not use escape_latex for setting_name in filename
        plot_filename = os.path.join(output_dir, f'{setting_name}_boxplot_{var.replace(" ", "_")}.png')
        plt.savefig(plot_filename); plt.close()
        print(f"  Saved box plot: {os.path.basename(plot_filename)}")

    print(f"\nGenerating Statistical Comparison Tables for {setting_name}...")
    higher_is_better_map = {
        'Total_Episode_Reward': True, 'Step_Reward': True,
        'Wastage_Cost': False, 'Lost_Sales_Cost': False, 'Holding_Cost': False,
        'Avg_InvLevel_All_Items': False,
    }
    for item_col in get_item_inv_cols(combined_df):
        if item_col not in higher_is_better_map: higher_is_better_map[item_col] = False

    for var in variables_to_analyze:
        if var not in combined_df.columns:
            print(f"  Variable '{var}' not found in combined data for stats table. Skipping.")
            continue

        methods = combined_df['Method'].unique()
        comparison_matrix = pd.DataFrame(index=methods, columns=methods, dtype=str)
        np.fill_diagonal(comparison_matrix.values, '-')

        data_for_var_by_method = {}
        if var == 'Total_Episode_Reward':
            unique_ep_rewards = combined_df[['Method', 'Seed', 'Episode', 'Total_Episode_Reward']].drop_duplicates()
            for method in methods:
                data_for_var_by_method[method] = unique_ep_rewards[unique_ep_rewards['Method'] == method][var]
        else:
            for method in methods:
                data_for_var_by_method[method] = combined_df[combined_df['Method'] == method][var].dropna()
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method_A, method_B = methods[i], methods[j]
                data_A, data_B = data_for_var_by_method[method_A], data_for_var_by_method[method_B]

                if data_A.empty or data_B.empty or data_A.nunique() < 2 or data_B.nunique() < 2 :
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = "NA (data)"
                    continue
                
                try:
                    stat, p_value = mannwhitneyu(data_A, data_B, alternative='two-sided')
                    is_significant = p_value < alpha
                    
                    result_str = "NS"
                    if is_significant:
                        median_A, median_B = data_A.median(), data_B.median()
                        is_A_better = (median_A > median_B) if higher_is_better_map.get(var, False) else (median_A < median_B)
                        result_str = f"{escape_latex(method_A) if is_A_better else escape_latex(method_B)} **" # Escape methods for table
                        
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = result_str
                except ValueError as e:
                    print(f"    Stat test error for {var} ({method_A} vs {method_B}): {e}")
                    comparison_matrix.loc[method_A, method_B] = comparison_matrix.loc[method_B, method_A] = "Error"

        print(f"\n  Statistical Comparison Matrix for: {var} (Setting: {escape_latex(setting_name)})")
        print(f"  (Higher value is better for '{var}': {higher_is_better_map.get(var, 'N/A (assuming lower)')})")
        # For console output and CSV, use original method names
        temp_comparison_matrix_for_output = comparison_matrix.copy()
        for r_idx, r_val in enumerate(methods): temp_comparison_matrix_for_output.rename(index={r_val: escape_latex(r_val)}, inplace=True)
        for c_idx, c_val in enumerate(methods): temp_comparison_matrix_for_output.rename(columns={c_val: escape_latex(c_val)}, inplace=True)
        print(temp_comparison_matrix_for_output.to_string())
        
        # CORRECTED FILENAME GENERATION: Do not use escape_latex for setting_name in filename
        matrix_filename = os.path.join(output_dir, f'{setting_name}_stat_matrix_{var.replace(" ", "_")}.csv')
        # Save original (non-escaped methods) comparison_matrix to CSV for easier machine parsing
        comparison_matrix.to_csv(matrix_filename)
        print(f"  Saved stat matrix: {os.path.basename(matrix_filename)}")

    # --- Calculate averages for summary table ---
    setting_summary_rows = []
    for method_name_orig, group_df in combined_df.groupby('Method'):
        # Use original method name for keys, escape later for display if needed
        summary_row = {'Setting': setting_name, 'Method': method_name_orig}
        for original_metric_name, display_metric_name in SUMMARY_METRICS_CONFIG.items():
            if original_metric_name == 'Total_Episode_Reward':
                if 'Total_Episode_Reward' in group_df.columns:
                    ep_rewards_method = group_df[['Episode', 'Seed', 'Total_Episode_Reward']].drop_duplicates()
                    summary_row[display_metric_name] = ep_rewards_method['Total_Episode_Reward'].mean()
                else:
                    summary_row[display_metric_name] = np.nan
            elif original_metric_name in group_df.columns:
                summary_row[display_metric_name] = group_df[original_metric_name].mean()
            else:
                summary_row[display_metric_name] = np.nan
        setting_summary_rows.append(summary_row)
    
    return setting_summary_rows


def generate_summary_text_file(all_summary_data, column_order, filepath):
    if not all_summary_data:
        print("No summary data to write to text file.")
        return

    df_summary = pd.DataFrame(all_summary_data)
    
    for col in column_order:
        if col not in df_summary.columns:
            df_summary[col] = np.nan
    df_summary = df_summary[column_order] 

    # Create a display version for printing, with LaTeX escapes for relevant text fields
    df_display_summary = df_summary.copy()
    df_display_summary['Setting'] = df_display_summary['Setting'].apply(escape_latex)
    df_display_summary['Method'] = df_display_summary['Method'].apply(escape_latex)

    for col in df_display_summary.columns:
        if df_display_summary[col].dtype == np.float64 or df_display_summary[col].dtype == np.int64:
            # Apply formatting after potential LaTeX escape on string columns
             df_display_summary[col] = df_display_summary[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
            
    # Save the original data (without LaTeX escapes in text) to CSV for machine readability
    df_summary_to_csv = df_summary.copy()
    for col in df_summary_to_csv.columns: # Format numerics for CSV too
        if df_summary_to_csv[col].dtype == np.float64 or df_summary_to_csv[col].dtype == np.int64:
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
        if col not in df_summary.columns:
            df_summary[col] = np.nan 
    df_summary = df_summary[column_order] 

    num_metric_cols = len(column_order) - 2 
    col_spec = "ll" + "r" * num_metric_cols 

    latex_string = "\\begin{table}[htbp]\n"
    latex_string += "  \\centering\n"
    latex_string += "  \\caption{Summary of Average Performance Metrics Across Settings and Methods}\n"
    latex_string += "  \\label{tab:summary_metrics_auto}\n"
    latex_string += f"  \\begin{{tabular}}{{{col_spec}}}\n"
    latex_string += "    \\toprule\n"
    
    header = " & ".join([escape_latex(col) for col in df_summary.columns]) + " \\\\\n"
    latex_string += f"    {header}"
    latex_string += "    \\midrule\n"

    for _, row in df_summary.iterrows():
        row_values = []
        for col_idx, col_name in enumerate(df_summary.columns):
            val = row[col_name]
            if isinstance(val, float) or isinstance(val, np.float64):
                row_values.append(f"{val:.2f}" if pd.notnull(val) else "-")
            else: # Setting and Method columns
                row_values.append(escape_latex(str(val)))
        latex_string += "    " + " & ".join(row_values) + " \\\\\n"

    latex_string += "    \\bottomrule\n"
    latex_string += "  \\end{tabular}\n"
    latex_string += "\\end{table}\n"
    latex_string += "% Note: Ensure the 'booktabs' package (\\usepackage{booktabs}) is included in your LaTeX preamble.\n"

    with open(filepath, 'w', encoding='utf-8') as f: # Specify encoding
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
        with open(results_txt_main_log_path, 'w', encoding='utf-8') as f_out: # Specify encoding
            sys.stdout = Tee(original_stdout, f_out) 

            print(f"Analysis Run: {timestamp}\nLOG_DIR: {os.path.abspath(LOG_DIR)}\nOutput DIR: {os.path.abspath(current_run_output_dir)}\n" + "-"*50)
            
            all_csv_in_log = glob.glob(os.path.join(LOG_DIR, '*.csv'))
            print(f"Diagnostic: Found {len(all_csv_in_log)} CSVs in LOG_DIR ('{os.path.abspath(LOG_DIR)}'). First 3: {all_csv_in_log[:3] if all_csv_in_log else 'None'}\n" + "-"*50)

            # Discover settings dynamically or define explicitly
            discovered_settings = set()
            if all_csv_in_log:
                for f_path in all_csv_in_log:
                    details = parse_experiment_details_from_filename(f_path)
                    if details:
                        discovered_settings.add(details['setting'])
            
            if not discovered_settings:
                 print("No settings discovered from CSV filenames. Using default 'setting_1'.")
                 settings_to_analyze = ["setting_1"] # Fallback
            else:
                 settings_to_analyze = sorted(list(discovered_settings))
                 if "default_setting" in settings_to_analyze and len(settings_to_analyze) > 1:
                     settings_to_analyze.remove("default_setting")
                     settings_to_analyze.append("default_setting") # Ensure default_setting (if present) is last

            print(f"Analyzing settings: {settings_to_analyze}")
            
            variables_for_analysis = [
                'Total_Episode_Reward', 
                'Wastage_Cost', 
                'Lost_Sales_Cost', 
                'Holding_Cost', 
                'Avg_InvLevel_All_Items',
                'Step_Reward' 
            ]

            all_settings_summary_data = []
            for setting in settings_to_analyze:
                summary_data_for_setting = perform_setting_analysis(setting, variables_for_analysis, current_run_output_dir)
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
                if f_tee != original_stdout and hasattr(f_tee, 'close') and not f_tee.closed:
                    f_tee.close()
        sys.stdout = original_stdout 
        print(f"\nFull analysis log saved to: {os.path.abspath(results_txt_main_log_path)}")
        if 'all_settings_summary_data' in locals() and all_settings_summary_data: # Check if defined
            print(f"Summary table (CSV) saved to: {os.path.abspath(summary_text_table_path)}")
            print(f"Summary table (LaTeX) saved to: {os.path.abspath(summary_latex_table_path)}")
        print(f"All plots and detailed stat matrices saved in: {os.path.abspath(current_run_output_dir)}")