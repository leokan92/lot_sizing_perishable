# src/agents/BSPEWAgent.py
from src.agents.BaseStockPolicyAgent import BaseStockPolicyAgent
import numpy as np
import sys

class BSPEWAgent(BaseStockPolicyAgent):
    def __init__(self, env,
                 waste_estimation_method: str = "deterministic_simulation", # Default to this new approach
                 waste_horizon_review_periods: int = 1,
                 num_ew_demand_sim_paths: int = 30, # Number of MC paths for demand estimation
                 **kwargs):

        self.waste_estimation_method = waste_estimation_method
        self.waste_horizon_review_periods = waste_horizon_review_periods
        self.num_ew_demand_sim_paths = num_ew_demand_sim_paths

        ew_rng_seed = env._initial_seed + 78901 if env._initial_seed is not None else np.random.randint(1e9)
        self.ew_sim_rng = np.random.default_rng(ew_rng_seed)


        super().__init__(env, **kwargs) # Call parent constructor

        if self.waste_estimation_method == "closed_form_approx":
            # This print statement remains accurate as it describes the general approach for mu estimation.
            print(f"Using closed-form approximation for EW (will estimate mu via MC with {self.num_ew_demand_sim_paths} paths). Assumes FIFO policy.")
        elif self.waste_estimation_method == "deterministic_simulation":
            print(f"Using deterministic simulation for EW (will estimate future daily demands via MC with {self.num_ew_demand_sim_paths} paths).")
        else:
            print(f"Warning: Unknown waste_estimation_method '{self.waste_estimation_method}'. Defaulting to deterministic_simulation.")
            self.waste_estimation_method = "deterministic_simulation"

    def _get_static_mean_demand_for_item(self, item_idx: int) -> float: # For closed-form fallback (if MC estimation not used, though currently it is)
        try:
            return self.env.stoch_model.mean_demands[item_idx]
        except AttributeError:
            try:
                return self.env.settings['mean_demands_per_item'][item_idx]
            except (KeyError, TypeError, IndexError):
                print(f"CRITICAL (Closed-Form EW): Static mean demand not found for item {item_idx}. Using 1.0.")
                return 1.0

    # _get_fifo_fraction_for_item is removed as it's no longer needed with a hardcoded FIFO policy for closed-form.

    def _estimate_future_daily_demands_mc(self, item_idx: int, horizon: int) -> np.ndarray:
        """
        Estimates daily demands for a specific item over a given horizon
        using Monte Carlo simulation based on the environment's stochastic demand model.
        Returns:
            np.ndarray: Array of shape (horizon,) with estimated mean daily demand for each future day.
        """
        if horizon <= 0 or self.num_ew_demand_sim_paths <= 0:
            return np.zeros(horizon) # Return array of zeros of correct shape if no sim

        all_simulated_demands_for_item = np.zeros((self.num_ew_demand_sim_paths, horizon))

        original_stoch_model_rng = None
        stoch_model_had_rng_attr = hasattr(self.env.stoch_model, 'rng')

        if stoch_model_had_rng_attr:
            original_stoch_model_rng = self.env.stoch_model.rng
            self.env.stoch_model.rng = self.ew_sim_rng # Use agent's dedicated EW RNG
        else:
            # If the stochastic model doesn't have an 'rng' attribute that we can swap,
            # we proceed, but multiple calls to generate_scenario might not be independent
            # in the way intended for EW simulation paths if it uses a global or unmanaged RNG.
            # This is a pre-existing condition/behavior if stoch_model doesn't have 'rng'.
            pass


        for i_path in range(self.num_ew_demand_sim_paths):
            # Generate a demand scenario for ALL items for the required horizon
            # The stoch_model.generate_scenario should return (n_items, n_time_steps)
            try:
                full_scenario = self.env.stoch_model.generate_scenario(n_time_steps=horizon)
                if full_scenario.ndim == 2 and full_scenario.shape[0] == self.env.n_items and full_scenario.shape[1] == horizon:
                    all_simulated_demands_for_item[i_path, :] = full_scenario[item_idx, :]
                elif full_scenario.ndim == 1 and self.env.n_items == 1 and full_scenario.shape[0] == horizon: # Handle case for single item, scenario is 1D
                    all_simulated_demands_for_item[i_path, :] = full_scenario[:]
                else:
                    print(f"Warning (EW Demand MC): stoch_model.generate_scenario returned unexpected shape "
                          f"{full_scenario.shape}, expected ({self.env.n_items}, {horizon}) or ({horizon},) for single item. Using zeros for path {i_path}.")
                    all_simulated_demands_for_item[i_path, :] = np.zeros(horizon)
            except Exception as e:
                print(f"Error (EW Demand MC): during stoch_model.generate_scenario: {e}. Using zeros for path {i_path}.")
                all_simulated_demands_for_item[i_path, :] = np.zeros(horizon)


        # Restore original RNG to stochastic model if we changed it
        if stoch_model_had_rng_attr and original_stoch_model_rng is not None:
            self.env.stoch_model.rng = original_stoch_model_rng

        # Calculate the average demand for each day in the horizon
        avg_daily_demands = np.mean(all_simulated_demands_for_item, axis=0)
        return avg_daily_demands


    def _calculate_ew_deterministic_simulation(self, item_idx: int, chosen_supplier_lead_time: int) -> float:
        """
        Estimates expected waste by simulating inventory evolution over a horizon,
        using daily demands estimated via Monte Carlo. This method inherently uses FIFO for demand satisfaction.
        """
        current_inventory_sim = self.env.inventory_age[item_idx, :].copy() # Simulate on a copy
        max_age_M = self.env.max_age

        sim_horizon = self.waste_horizon_review_periods + chosen_supplier_lead_time - 1
        if sim_horizon <= 0:
            return 0.0

        # --- Estimate future daily demands using Monte Carlo ---
        future_daily_demands_est = self._estimate_future_daily_demands_mc(item_idx, sim_horizon)
        # ---

        total_simulated_waste = 0

        for t_sim in range(sim_horizon):
            # Get the estimated demand for the current simulated day
            demand_for_this_sim_day = future_daily_demands_est[t_sim]
            # Since demand can be fractional from averaging, round for inventory operations
            demand_for_this_sim_day_int = np.round(demand_for_this_sim_day).astype(int)
            if demand_for_this_sim_day_int < 0: demand_for_this_sim_day_int = 0


            # 1. Age inventory for this simulation step
            inventory_at_start_of_aging_sim = current_inventory_sim.copy()
            next_step_inventory_sim = np.zeros_like(current_inventory_sim)
            simulated_waste_this_step = 0

            for age_idx_sim in range(max_age_M):
                n_items_in_bin_sim = inventory_at_start_of_aging_sim[age_idx_sim]
                if n_items_in_bin_sim == 0:
                    continue

                # Expiration logic based on shelf_life_cdf
                cdf_age_plus_1 = self.env.shelf_life_cdf[item_idx, age_idx_sim + 1] if (age_idx_sim + 1) < max_age_M else 1.0
                cdf_age = self.env.shelf_life_cdf[item_idx, age_idx_sim]
                prob_survival_up_to_age = 1.0 - cdf_age
                
                if prob_survival_up_to_age <= 1e-9: # Effectively zero survival probability
                    prob_expire_in_next_step_sim = 1.0
                else:
                    prob_expire_in_next_step_sim = np.clip((cdf_age_plus_1 - cdf_age) / prob_survival_up_to_age, 0.0, 1.0)
                
                wasted_count_sim = np.round(n_items_in_bin_sim * prob_expire_in_next_step_sim).astype(int)
                simulated_waste_this_step += wasted_count_sim
                survivors_count_sim = n_items_in_bin_sim - wasted_count_sim

                if age_idx_sim < max_age_M - 1 and survivors_count_sim > 0:
                    next_step_inventory_sim[age_idx_sim + 1] += survivors_count_sim
            
            current_inventory_sim = next_step_inventory_sim
            total_simulated_waste += simulated_waste_this_step

            # 2. Satisfy estimated demand (FIFO from current_inventory_sim)
            demand_to_satisfy_sim = demand_for_this_sim_day_int
            for age_idx_sim in range(max_age_M - 1, -1, -1): # Oldest first for FIFO
                if demand_to_satisfy_sim <= 0: break
                available_in_bin_sim = current_inventory_sim[age_idx_sim]
                if available_in_bin_sim <= 0: continue
                
                fulfilled_from_bin_sim = min(available_in_bin_sim, demand_to_satisfy_sim)
                current_inventory_sim[age_idx_sim] -= fulfilled_from_bin_sim
                demand_to_satisfy_sim -= fulfilled_from_bin_sim
        
        return float(total_simulated_waste)

    def _calculate_ew_closed_form_approx(self, item_idx: int, chosen_supplier_lead_time: int) -> float:
        """
        Estimates expected waste using a closed-form approximation, strictly assuming FIFO.
        """
        current_inventory_by_age = self.env.inventory_age[item_idx, :].copy()
        max_shelf_life_M = self.env.max_age
        sim_horizon = self.waste_horizon_review_periods + chosen_supplier_lead_time - 1
        if sim_horizon <= 0: return 0.0

        future_daily_demands_est = self._estimate_future_daily_demands_mc(item_idx, sim_horizon)
        if future_daily_demands_est.size == 0 :
            mean_demand_mu_for_horizon = 0.0
        else:
            mean_demand_mu_for_horizon = np.mean(future_daily_demands_est) # Avg per-step demand over horizon

        sum_old_stock = 0
        # Items that will expire within sim_horizon if not consumed
        # Age k now will be k + sim_horizon. Expires if k + sim_horizon >= max_shelf_life_M
        # So, k >= max_shelf_life_M - sim_horizon
        start_age_for_old_stock = max(0, int(np.ceil(max_shelf_life_M - sim_horizon))) # Use ceil to be conservative: if M-h = 3.1, items of age 4 are old.
                                                                                     # Or rather, items that are AT LEAST this age.
                                                                                     # if max_shelf_life_M = 5, sim_horizon = 2. Old stock starts at age 5-2 = 3. (ages 3, 4)
        # Iterate up to max_shelf_life_M - 1 as ages are 0-indexed
        for age_k in range(start_age_for_old_stock, int(max_shelf_life_M)):
            if age_k < current_inventory_by_age.shape[0]: # Boundary check
                 sum_old_stock += current_inventory_by_age[age_k]


        # Closed-form uses total demand over horizon
        total_demand_for_horizon_est = mean_demand_mu_for_horizon * sim_horizon

        # FIFO EW: Waste is old stock that is not consumed by total demand.
        ew_fifo = max(0, sum_old_stock - total_demand_for_horizon_est)
        
        return float(ew_fifo)


    def _get_action_from_policy(self, bsp_policy_matrix: np.ndarray) -> np.ndarray:
        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        current_inventory_level = self.env.inventory_level
        outstanding_orders = self._calculate_outstanding_orders() 

        for i in range(self.env.n_items):
            chosen_supplier_indices = np.where(bsp_policy_matrix[i, :] > 0)[0]

            if len(chosen_supplier_indices) > 0:
                s = chosen_supplier_indices[0] # Assuming single sourcing or taking the first policy-defined supplier
                target_level = bsp_policy_matrix[i, s]
                on_hand = current_inventory_level[i]
                outstanding = outstanding_orders[i]
                ew_item_i = 0.0
                chosen_supplier_lead_time = int(self.env.lead_times[i,s])

                if self.waste_estimation_method == "closed_form_approx":
                    ew_item_i = self._calculate_ew_closed_form_approx(i, chosen_supplier_lead_time)
                elif self.waste_estimation_method == "deterministic_simulation":
                    ew_item_i = self._calculate_ew_deterministic_simulation(i, chosen_supplier_lead_time)
                else: 
                    # This case should ideally be caught by __init__ default, but defensive here.
                    print(f"Warning: Unknown EW method '{self.waste_estimation_method}' in _get_action_from_policy. Using 0 EW for item {i}.")

                inventory_position = on_hand + outstanding
                order_qty_bsp_ew = target_level - inventory_position + ew_item_i
                action[i, s] = float(max(0, order_qty_bsp_ew))
        return action