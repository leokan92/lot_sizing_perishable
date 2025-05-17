import os
import csv
from datetime import datetime
import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # This warning can be printed by the agent or runner if logging is attempted without pandas
    # print("Warning: pandas library not found. Detailed CSV logging will be basic if enabled.")

class SimulationLogger:
    def __init__(self,
                 log_dir: str,
                 experiment_name: str, # e.g., env_agent_seed
                 log_step_details: bool = True,
                 log_actions: bool = False,
                 n_items: int = 0,
                 n_suppliers: int = 0,
                 item_names: list = None,
                 supplier_names: list = None):

        self.log_step_details_enabled = log_step_details
        if not self.log_step_details_enabled:
            return

        self.log_dir = os.path.abspath(log_dir)
        self.experiment_name = experiment_name
        self.log_actions = log_actions
        self.n_items = n_items
        self.n_suppliers = n_suppliers

        self.item_names = item_names if item_names else [f"Item{i}" for i in range(n_items)]
        self.supplier_names = supplier_names if supplier_names else [f"Sup{s}" for s in range(n_suppliers)]

        os.makedirs(self.log_dir, exist_ok=True)
        # Filename uses experiment_name, which should be unique per run (env_agent_seed)
        self.log_file_path = os.path.join(self.log_dir, f"{self.experiment_name}_sim_details.csv")

        self.all_episodes_data = [] # Stores dicts for all steps of all episodes
        self.current_episode_num = -1

        self.header_generated = False
        self.header = []
        self._generate_header() # Generate header at initialization

    def _generate_header(self):
        if self.header_generated:
            return

        base_header = [
            "Episode", "Step", "Step_Reward",
            "Purchase_Cost", "Fixed_Order_Cost", "Holding_Cost",
            "Lost_Sales_Cost", "Wastage_Cost"
        ]
        for i_name in self.item_names:
            base_header.append(f"{i_name}_InvLevel")
            base_header.append(f"{i_name}_DemandUnits")
            base_header.append(f"{i_name}_WastageUnits")
            base_header.append(f"{i_name}_ArrivalsUnits")

        if self.log_actions:
            for i_idx, i_name in enumerate(self.item_names):
                for s_idx, s_name in enumerate(self.supplier_names):
                    base_header.append(f"{i_name}_{s_name}_OrderQty")
        
        self.header = base_header
        self.header_generated = True

    def start_episode(self, episode_num: int):
        if not self.log_step_details_enabled:
            return
        self.current_episode_num = episode_num

    def log_step(self, step_num: int, reward: float, info: dict, action: np.ndarray = None):
        if not self.log_step_details_enabled:
            return
        
        log_entry = {
            "Episode": self.current_episode_num,
            "Step": step_num,
            "Step_Reward": reward,
            "Purchase_Cost": info.get('purchase_costs', 0.0),
            "Fixed_Order_Cost": info.get('fixed_order_costs', 0.0),
            "Holding_Cost": info.get('holding_costs', 0.0),
            "Lost_Sales_Cost": info.get('lost_sales_costs', 0.0),
            "Wastage_Cost": info.get('wastage_costs', 0.0)
        }

        inventory_levels = info.get('inventory_level', np.zeros(self.n_items))
        demand_units = info.get('demand_units', np.zeros(self.n_items))
        wastage_units = info.get('wastage_units', np.zeros(self.n_items))
        arrivals_units = info.get('arrivals_units', np.zeros(self.n_items))

        for i_idx, i_name in enumerate(self.item_names):
            log_entry[f"{i_name}_InvLevel"] = inventory_levels[i_idx] if i_idx < len(inventory_levels) else 0
            log_entry[f"{i_name}_DemandUnits"] = demand_units[i_idx] if i_idx < len(demand_units) else 0
            log_entry[f"{i_name}_WastageUnits"] = wastage_units[i_idx] if i_idx < len(wastage_units) else 0
            log_entry[f"{i_name}_ArrivalsUnits"] = arrivals_units[i_idx] if i_idx < len(arrivals_units) else 0

        if self.log_actions and action is not None:
            order_quantities = np.maximum(0, np.round(action)).astype(int)
            for i_idx, i_name in enumerate(self.item_names):
                for s_idx, s_name in enumerate(self.supplier_names):
                    if i_idx < order_quantities.shape[0] and s_idx < order_quantities.shape[1]:
                         log_entry[f"{i_name}_{s_name}_OrderQty"] = order_quantities[i_idx, s_idx]
                    else: # Should not happen if n_items, n_suppliers are correctly passed
                         log_entry[f"{i_name}_{s_name}_OrderQty"] = 0
        
        self.all_episodes_data.append(log_entry)

    def end_episode(self):
        if not self.log_step_details_enabled:
            return
        # No specific action needed here as data is collected in self.all_episodes_data

    def finalize_logs(self):
        if not self.log_step_details_enabled or not self.all_episodes_data:
            if self.log_step_details_enabled and not self.all_episodes_data: # Log attempt but no data
                print(f"SimulationLogger: No data recorded for {self.experiment_name}. Log file '{self.log_file_path}' not created.")
            return

        print(f"SimulationLogger: Saving detailed simulation logs to {self.log_file_path}")
        if PANDAS_AVAILABLE:
            try:
                df = pd.DataFrame(self.all_episodes_data)
                # Ensure columns are in the order of self.header and all header columns are present
                df = df.reindex(columns=self.header) 
                df.to_csv(self.log_file_path, index=False, float_format='%.4f')
            except Exception as e:
                print(f"SimulationLogger: Error saving logs with pandas: {e}. Falling back to basic CSV write.")
                self._save_with_csv_module() 
        else:
            print("SimulationLogger: pandas library not found. CSV logging will be basic.")
            self._save_with_csv_module()
        
        self.all_episodes_data = [] # Clear data after saving

    def _save_with_csv_module(self):
        if not self.all_episodes_data or not self.header:
            return
        
        try:
            with open(self.log_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.header, restval='NA')
                writer.writeheader()
                # DictWriter handles rows that might be missing some keys specified in fieldnames (fills with restval)
                # and ignores extra keys in rows not in fieldnames if extrasaction='ignore' (default raises error).
                # Our log_entry should contain keys that are a subset of or equal to self.header.
                writer.writerows(self.all_episodes_data)
        except IOError as e:
            print(f"SimulationLogger: Error writing logs with csv module: {e}")
        except Exception as ex: 
            print(f"SimulationLogger: An unexpected error occurred during basic CSV writing: {ex}")