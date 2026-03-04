from src.agents.BaseStockPolicyAgent import BaseStockPolicyAgent
import numpy as np
import sys

class BSPEWAgent(BaseStockPolicyAgent):
    def __init__(self, env,
                 waste_estimation_method: str = "deterministic_simulation",
                 waste_horizon_review_periods: int = 1,
                 num_ew_demand_sim_paths: int = 30,
                 **kwargs):

        self.waste_estimation_method = waste_estimation_method
        self.waste_horizon_review_periods = waste_horizon_review_periods
        self.num_ew_demand_sim_paths = num_ew_demand_sim_paths

        ew_rng_seed = env._initial_seed + 78901 if env._initial_seed is not None else np.random.randint(1e9)
        self.ew_sim_rng = np.random.default_rng(ew_rng_seed)

        super().__init__(env, **kwargs)

        if self.waste_estimation_method == "closed_form_approx":
            print(f"Using closed-form approximation for EW (will estimate mu via MC with {self.num_ew_demand_sim_paths} paths). Assumes FIFO policy.")
        elif self.waste_estimation_method == "deterministic_simulation":
            print(f"Using deterministic simulation for EW (will estimate future daily demands via MC with {self.num_ew_demand_sim_paths} paths). Using float precision for internal simulation.") # Updated print
        else:
            print(f"Warning: Unknown waste_estimation_method '{self.waste_estimation_method}'. Defaulting to deterministic_simulation.")
            self.waste_estimation_method = "deterministic_simulation"

    def _get_static_mean_demand_for_item(self, item_idx: int) -> float:
        try:
            return self.env.stoch_model.mean_demands[item_idx]
        except AttributeError:
            try:
                return self.env.settings['mean_demands_per_item'][item_idx]
            except (KeyError, TypeError, IndexError):
                print(f"CRITICAL (Closed-Form EW): Static mean demand not found for item {item_idx}. Using 1.0.")
                return 1.0

    def _estimate_future_daily_demands_mc(self, item_idx: int, horizon: int) -> np.ndarray:
        if horizon <= 0 or self.num_ew_demand_sim_paths <= 0:
            return np.zeros(horizon)

        all_simulated_demands_for_item = np.zeros((self.num_ew_demand_sim_paths, horizon))
        original_stoch_model_rng = None
        stoch_model_had_rng_attr = hasattr(self.env.stoch_model, 'rng')

        if stoch_model_had_rng_attr:
            original_stoch_model_rng = self.env.stoch_model.rng
            self.env.stoch_model.rng = self.ew_sim_rng
        else:
            pass

        for i_path in range(self.num_ew_demand_sim_paths):
            try:
                full_scenario = self.env.stoch_model.generate_scenario(n_time_steps=horizon)
                if full_scenario.ndim == 2 and full_scenario.shape[0] == self.env.n_items and full_scenario.shape[1] == horizon:
                    all_simulated_demands_for_item[i_path, :] = full_scenario[item_idx, :]
                elif full_scenario.ndim == 1 and self.env.n_items == 1 and full_scenario.shape[0] == horizon:
                    all_simulated_demands_for_item[i_path, :] = full_scenario[:]
                else:
                    print(f"Warning (EW Demand MC): stoch_model.generate_scenario returned unexpected shape "
                          f"{full_scenario.shape}, expected ({self.env.n_items}, {horizon}) or ({horizon},) for single item. Using zeros for path {i_path}.")
                    all_simulated_demands_for_item[i_path, :] = np.zeros(horizon)
            except Exception as e:
                print(f"Error (EW Demand MC): during stoch_model.generate_scenario: {e}. Using zeros for path {i_path}.")
                all_simulated_demands_for_item[i_path, :] = np.zeros(horizon)

        if stoch_model_had_rng_attr and original_stoch_model_rng is not None:
            self.env.stoch_model.rng = original_stoch_model_rng

        avg_daily_demands = np.mean(all_simulated_demands_for_item, axis=0)
        return avg_daily_demands

    def _calculate_ew_deterministic_simulation(self, item_idx: int, chosen_supplier_lead_time: int) -> float:
        """
        Estimates expected waste by simulating inventory evolution over a horizon,
        using daily demands estimated via Monte Carlo. This method inherently uses FIFO for demand satisfaction.
        MODIFIED: Uses floating-point numbers throughout the internal simulation to avoid rounding bias.
        """
        # Ensure current_inventory_sim is float for the simulation
        current_inventory_sim = self.env.inventory_age[item_idx, :].astype(float) # MODIFIED: Ensure float
        max_age_M = self.env.max_age

        sim_horizon = self.waste_horizon_review_periods + chosen_supplier_lead_time - 1
        if sim_horizon <= 0:
            return 0.0

        future_daily_demands_est = self._estimate_future_daily_demands_mc(item_idx, sim_horizon)
        
        total_simulated_waste_float = 0.0 # MODIFIED: Init as float

        for t_sim in range(sim_horizon):
            demand_for_this_sim_day_float = future_daily_demands_est[t_sim] # Is already float
            if demand_for_this_sim_day_float < 0: # Ensure non-negative demand
                demand_for_this_sim_day_float = 0.0

            # 1. Age inventory for this simulation step
            inventory_at_start_of_aging_sim = current_inventory_sim.copy() # Is float
            next_step_inventory_sim = np.zeros_like(current_inventory_sim, dtype=float) # MODIFIED: Ensure float
            simulated_waste_this_step_float = 0.0 # MODIFIED: Init as float

            for age_idx_sim in range(max_age_M):
                n_items_in_bin_sim_float = inventory_at_start_of_aging_sim[age_idx_sim] # Is float
                if n_items_in_bin_sim_float <= 1e-9: # Effectively zero items
                    continue

                cdf_age_plus_1 = self.env.shelf_life_cdf[item_idx, age_idx_sim + 1] if (age_idx_sim + 1) < max_age_M else 1.0
                cdf_age = self.env.shelf_life_cdf[item_idx, age_idx_sim]
                prob_survival_up_to_age = 1.0 - cdf_age
                
                if prob_survival_up_to_age <= 1e-9:
                    prob_expire_in_next_step_sim = 1.0
                else:
                    prob_expire_in_next_step_sim = np.clip((cdf_age_plus_1 - cdf_age) / prob_survival_up_to_age, 0.0, 1.0)
                
                # MODIFIED: Calculate waste as float, no rounding
                wasted_count_sim_float = n_items_in_bin_sim_float * prob_expire_in_next_step_sim
                simulated_waste_this_step_float += wasted_count_sim_float
                survivors_count_sim_float = n_items_in_bin_sim_float - wasted_count_sim_float

                if age_idx_sim < max_age_M - 1 and survivors_count_sim_float > 1e-9: # Check against small float
                    next_step_inventory_sim[age_idx_sim + 1] += survivors_count_sim_float
            
            current_inventory_sim = next_step_inventory_sim # current_inventory_sim is now float
            total_simulated_waste_float += simulated_waste_this_step_float

            # 2. Satisfy estimated demand (FIFO from current_inventory_sim) using floats
            demand_to_satisfy_sim_float = demand_for_this_sim_day_float # MODIFIED: Use float demand
            for age_idx_sim in range(max_age_M - 1, -1, -1): # Oldest first for FIFO
                if demand_to_satisfy_sim_float <= 1e-9: break # MODIFIED: Check against small float
                
                available_in_bin_sim_float = current_inventory_sim[age_idx_sim] # Is float
                if available_in_bin_sim_float <= 1e-9: continue # MODIFIED: Check against small float
                
                fulfilled_from_bin_sim_float = min(available_in_bin_sim_float, demand_to_satisfy_sim_float) # MODIFIED
                current_inventory_sim[age_idx_sim] -= fulfilled_from_bin_sim_float
                demand_to_satisfy_sim_float -= fulfilled_from_bin_sim_float
        
        return float(total_simulated_waste_float) # Already float, float() is harmless

    def _calculate_ew_closed_form_approx(self, item_idx: int, chosen_supplier_lead_time: int) -> float:
        """
        Estimates expected waste using a closed-form approximation, strictly assuming FIFO.
        """
        # Ensure current_inventory_by_age is float
        current_inventory_by_age = self.env.inventory_age[item_idx, :].astype(float) # MODIFIED: Ensure float
        max_shelf_life_M = self.env.max_age
        sim_horizon = self.waste_horizon_review_periods + chosen_supplier_lead_time - 1
        if sim_horizon <= 0: return 0.0

        future_daily_demands_est = self._estimate_future_daily_demands_mc(item_idx, sim_horizon)
        if future_daily_demands_est.size == 0 :
            mean_demand_mu_for_horizon = 0.0
        else:
            mean_demand_mu_for_horizon = np.mean(future_daily_demands_est)

        sum_old_stock_float = 0.0 # MODIFIED: Ensure float
        start_age_for_old_stock = max(0, int(np.ceil(max_shelf_life_M - sim_horizon)))
        
        for age_k in range(start_age_for_old_stock, int(max_shelf_life_M)):
            if age_k < current_inventory_by_age.shape[0]:
                 sum_old_stock_float += current_inventory_by_age[age_k] # current_inventory_by_age is float

        total_demand_for_horizon_est = mean_demand_mu_for_horizon * sim_horizon

        ew_fifo = max(0.0, sum_old_stock_float - total_demand_for_horizon_est) # MODIFIED: Ensure float comparison with 0.0
        
        return float(ew_fifo) # Already float

    def _get_action_from_policy(self, bsp_policy_matrix: np.ndarray) -> np.ndarray:
        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        current_inventory_level = self.env.inventory_level # This is sum of inventory_age, typically int/float
        outstanding_orders = self._calculate_outstanding_orders() # Returns int array

        for i in range(self.env.n_items):
            active_suppliers = np.where(bsp_policy_matrix[i, :] > 0)[0]
            if len(active_suppliers) == 0:
                continue

            # Total base stock target = sum of per-supplier levels
            total_target = float(np.sum(bsp_policy_matrix[i, active_suppliers]))
            on_hand = float(current_inventory_level[i])
            outstanding = float(outstanding_orders[i])
            inventory_position = on_hand + outstanding

            # Compute EW using weighted average lead time across active suppliers
            weights = np.array([float(bsp_policy_matrix[i, s]) for s in active_suppliers])
            weights /= weights.sum()
            avg_lead_time = int(np.round(np.sum([
                weights[k] * self.env.lead_times[i, s]
                for k, s in enumerate(active_suppliers)
            ])))
            avg_lead_time = max(1, avg_lead_time)

            ew_item_i = 0.0
            if self.waste_estimation_method == "closed_form_approx":
                ew_item_i = self._calculate_ew_closed_form_approx(i, avg_lead_time)
            elif self.waste_estimation_method == "deterministic_simulation":
                ew_item_i = self._calculate_ew_deterministic_simulation(i, avg_lead_time)
            else: 
                print(f"Warning: Unknown EW method '{self.waste_estimation_method}' in _get_action_from_policy. Using 0 EW for item {i}.")

            total_order = max(0.0, total_target - inventory_position + ew_item_i)

            # Split order proportionally across active suppliers
            if total_order > 0.0 and total_target > 0.0:
                for s in active_suppliers:
                    fraction = float(bsp_policy_matrix[i, s]) / total_target
                    action[i, s] = float(fraction * total_order)
        return action