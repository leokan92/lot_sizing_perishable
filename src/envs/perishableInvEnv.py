import numpy as np
from numpy.random import Generator, default_rng
import gymnasium as gym
from gymnasium import spaces
from src.scenarioManager.stochasticDemandModel import StochasticDemandModel # Example import



class PerishableInvEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, settings: dict, stochastic_model_settings: dict, seed=None): # stochastic_model_settings is stoch_model in main_runner
        super(PerishableInvEnv, self).__init__()
        self._initial_seed = seed
        self._current_seed_state = self._initial_seed
        self.settings = settings
        self.stoch_model = stochastic_model_settings # Corrected param name to match usage
        self.env_rng = default_rng(seed)

        # --- Basic cardinalities & Parameters ---
        self.T = settings['time_horizon']
        self.n_items = settings['n_items']
        self.n_suppliers = settings['n_suppliers']
        self.max_age = settings['max_age']
        self.max_inventory_level = np.array(settings['max_inventory_level'], dtype=int)
        self.item_supplier_matrix = np.array(settings['item_supplier_matrix'], dtype=int)
        self.unit_purchase_costs = np.array(settings['unit_purchase_costs'], dtype=float)
        self.fixed_order_costs = np.array(settings['fixed_order_costs'], dtype=float)
        self.lead_times = np.array(settings['lead_times'], dtype=int)
        self.prob_full_fulfillment = np.array(settings['prob_full_fulfillment'], dtype=float)
        self.partial_fulfillment_beta_alpha = np.array(settings['partial_fulfillment_beta_alpha'], dtype=float)
        self.partial_fulfillment_beta_beta = np.array(settings['partial_fulfillment_beta_beta'], dtype=float)
        self.shelf_life_cdf = np.array(settings['shelf_life_cdf'], dtype=float)
        self.holding_costs = np.array(settings['holding_costs'], dtype=float)
        self.lost_sales_costs = np.array(settings['lost_sales_costs'], dtype=float)
        self.initial_inventory_age = np.array(settings['initial_inventory_age'], dtype=int)
        self.max_items_for_wastage_draws = settings.get('max_items_for_wastage_draws',
                                                      int(np.max(self.max_inventory_level) * 1.5) + 50)

        # --- Calculate Minimum Purchase Cost per Item (used for initial value if needed) ---
        costs_with_inf = np.where(self.item_supplier_matrix == 1,
                                  self.unit_purchase_costs, np.inf)
        self.min_purchase_costs_per_item = np.min(costs_with_inf, axis=1)
        self.min_purchase_costs_per_item[np.isinf(self.min_purchase_costs_per_item)] = 0.0 # Default if no supplier

        # --- Handle Initial Inventory Value ---
        # CORRECTED: Get 'initial_inventory_value' from settings dictionary
        self.initial_inventory_value_setting = settings.get('initial_inventory_value', None)

        if self.initial_inventory_value_setting is None:
            # Default: Value initial inventory at minimum purchase cost
            self.initial_inventory_value = np.zeros_like(self.initial_inventory_age, dtype=float)
            for i in range(self.n_items):
                 self.initial_inventory_value[i, :] = self.initial_inventory_age[i, :] * self.min_purchase_costs_per_item[i]
            #print("Info: 'initial_inventory_value' not specified in settings. Initializing value using minimum purchase costs.")
        elif isinstance(self.initial_inventory_value_setting, (int, float)) and self.initial_inventory_value_setting == 0:
             self.initial_inventory_value = np.zeros_like(self.initial_inventory_age, dtype=float)
             #print("Info: 'initial_inventory_value' set to 0 in settings. Initializing inventory value to zero.")
        else:
            # Assume it's a numpy array compatible with initial_inventory_age
            try:
                 # Ensure it's a numpy array before checking shape
                 provided_value_array = np.array(self.initial_inventory_value_setting, dtype=float)
                 assert provided_value_array.shape == self.initial_inventory_age.shape, \
                     f"Shape mismatch: initial_inventory_value shape {provided_value_array.shape} and initial_inventory_age shape {self.initial_inventory_age.shape}"
                 self.initial_inventory_value = provided_value_array
                 print("Info: Using provided 'initial_inventory_value' from settings.")
            except Exception as e:
                 # Add more context to the error message
                 raise ValueError(f"Invalid 'initial_inventory_value' setting in config. Expected shape {self.initial_inventory_age.shape} or 0 or null. Error: {e}")


        # --- Validation ---
        assert self.prob_full_fulfillment.shape == (self.n_items, self.n_suppliers), \
            f"prob_full_fulfillment shape mismatch: expected {(self.n_items, self.n_suppliers)}, got {self.prob_full_fulfillment.shape}"
        # ... other validations ...

        # --- State variables & Pre-generated randomness ---
        self.current_step = 0
        # Inventory Quantity and Value
        # Initialize here, but they will be properly set in reset()
        self.inventory_age = np.zeros((self.n_items, self.max_age), dtype=int)
        self.inventory_value = np.zeros_like(self.inventory_age, dtype=float)
        self.inventory_level = np.zeros(self.n_items, dtype=int)

        self.order_history = []
        self.demand = np.zeros(self.n_items, dtype=int)
        self.wastage = np.zeros(self.n_items, dtype=int)
        self.last_step_costs = {}
        self.scenario_demand = np.zeros((self.n_items, self.T), dtype=int)
        self.fulfillment_uniform_draws = np.zeros((self.T, self.n_items, self.n_suppliers))
        self.fulfillment_beta_fractions = np.zeros((self.T, self.n_items, self.n_suppliers))
        self.wastage_uniform_draws = np.zeros((self.T, self.n_items, self.max_age, self.max_items_for_wastage_draws))

        # --- Attributes for Render Debugging ---
        self.last_step_arrivals = np.zeros(self.n_items, dtype=int)
        self.last_step_demand = np.zeros(self.n_items, dtype=int)

        # --- Gym spaces ---
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(self.n_items, self.n_suppliers), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'inventory_age': spaces.Box(low=0, high=np.inf, shape=(self.n_items, self.max_age), dtype=int),
            # 'inventory_value': spaces.Box(low=0, high=np.inf, shape=(self.n_items, self.max_age), dtype=float), # Optional: Add if agent needs value state
            'current_step': spaces.Discrete(self.T)
        })


    # --- generate_scenario_realization, _pregenerate_randomness remain the same ---
    def generate_scenario_realization(self):
        self.stoch_model.rng = self.env_rng
        self.scenario_demand = self.stoch_model.generate_scenario(n_time_steps=self.T).astype(int)

    def _pregenerate_randomness(self):
        self.fulfillment_uniform_draws = self.env_rng.random(size=(self.T, self.n_items, self.n_suppliers))
        valid_beta_params = (self.partial_fulfillment_beta_alpha > 0) & (self.partial_fulfillment_beta_beta > 0)
        self.fulfillment_beta_fractions = np.zeros((self.T, self.n_items, self.n_suppliers))
        for i in range(self.n_items):
            for s in range(self.n_suppliers):
                 if valid_beta_params[i, s]:
                     self.fulfillment_beta_fractions[:, i, s] = self.env_rng.beta(
                         a=self.partial_fulfillment_beta_alpha[i, s],
                         b=self.partial_fulfillment_beta_beta[i, s],
                         size=self.T
                     )
                 else:
                      self.fulfillment_beta_fractions[:, i, s] = 0.0
        self.wastage_uniform_draws = self.env_rng.random(
            size=(self.T, self.n_items, self.max_age, self.max_items_for_wastage_draws)
        )

    def _get_observation(self):
        obs = {
            'inventory_age': self.inventory_age.copy(),
            # 'inventory_value': self.inventory_value.copy(), # Add if defined in obs space
            'current_step': self.current_step
        }
        return obs
    
    # --- _get_outstanding_orders_state remains the same ---
    def _get_outstanding_orders_state(self):
        outstanding = np.zeros(self.n_items, dtype=int)
        max_lead_time = np.max(self.lead_times) if self.lead_times.size > 0 else 0
        for t_placed, i, s, qty_ordered in self.order_history:
             if t_placed + self.lead_times[i, s] > self.current_step :
                   outstanding[i] += qty_ordered
        return outstanding


    def _receive_arrivals(self):
        arrivals_today_qty = np.zeros(self.n_items, dtype=int)
        arrivals_today_value = np.zeros(self.n_items, dtype=float) # Track value
        remaining_order_history = []

        for order_details in self.order_history:
            t_placed, i, s, qty_ordered = order_details
            arrival_time = t_placed + self.lead_times[i, s]

            if arrival_time == self.current_step:
                fulfilled_qty = 0 # Default if order qty is 0 or randomness fails
                if qty_ordered > 0:
                    # Use Pre-generated Fulfillment Randomness
                    if t_placed < self.T:
                        prob_full = self.prob_full_fulfillment[i, s]
                        uniform_draw = self.fulfillment_uniform_draws[t_placed, i, s]

                        if uniform_draw <= prob_full:
                            fulfilled_qty = qty_ordered
                        else:
                            fulfillment_fraction = self.fulfillment_beta_fractions[t_placed, i, s]
                            fulfilled_qty = np.round(qty_ordered * fulfillment_fraction).astype(int)
                    else:
                        print(f"Warning: Accessing fulfillment randomness for t_placed={t_placed} >= T")
                        fulfilled_qty = 0

                if fulfilled_qty > 0:
                     purchase_cost = self.unit_purchase_costs[i, s] # Cost for this specific item/supplier
                     arrivals_today_qty[i] += fulfilled_qty
                     arrivals_today_value[i] += fulfilled_qty * purchase_cost # Add value

            elif arrival_time > self.current_step:
                remaining_order_history.append(order_details)

        self.order_history = remaining_order_history

        # Add arrivals to the newest age bin (age 0) for quantity and value
        if np.any(arrivals_today_qty > 0):
             self.inventory_age[:, 0] += arrivals_today_qty
             self.inventory_value[:, 0] += arrivals_today_value # Update value
             self.inventory_level = np.sum(self.inventory_age, axis=1)

        return arrivals_today_qty # Return just the quantity for info/render

    # --- _place_new_orders remains the same ---
    def _place_new_orders(self, action):
        order_quantities = np.maximum(0, np.round(action)).astype(int)
        purchase_cost_step = 0.0
        fixed_cost_step = 0.0
        supplier_used_this_step = np.zeros(self.n_suppliers, dtype=int)

        for i in range(self.n_items):
            for s in range(self.n_suppliers):
                if self.item_supplier_matrix[i, s] == 1 and order_quantities[i, s] > 0:
                    qty_ordered = order_quantities[i, s]
                    purchase_cost_step += self.unit_purchase_costs[i, s] * qty_ordered
                    supplier_used_this_step[s] = 1
                    self.order_history.append((self.current_step, i, s, qty_ordered))

        fixed_cost_step = np.sum(self.fixed_order_costs * supplier_used_this_step)
        return purchase_cost_step, fixed_cost_step


    def _satisfy_demand_and_calc_costs(self):
        lost_sales_cost_step = 0.0
        holding_cost_step = 0.0
        demand_to_satisfy = self.demand.copy()

        # --- Satisfy Demand (FIFO) & Update Value ---
        for i in range(self.n_items):
            for age_idx in range(self.max_age - 1, -1, -1): # This is subtracting from the oldest age bin first
                if demand_to_satisfy[i] <= 0: break

                available_in_bin = self.inventory_age[i, age_idx]
                if available_in_bin <= 0: continue # Skip empty bins

                value_in_bin = self.inventory_value[i, age_idx]
                # Calculate avg cost BEFORE fulfilling from bin
                avg_cost_in_bin = value_in_bin / available_in_bin if available_in_bin > 0 else 0

                fulfilled_from_bin = min(available_in_bin, demand_to_satisfy[i])

                if fulfilled_from_bin > 0:
                    # Update quantity
                    self.inventory_age[i, age_idx] -= fulfilled_from_bin
                    # Update value based on average cost of items sold
                    self.inventory_value[i, age_idx] -= fulfilled_from_bin * avg_cost_in_bin
                    # Ensure value doesn't go negative due to float precision
                    self.inventory_value[i, age_idx] = max(0, self.inventory_value[i, age_idx])
                    demand_to_satisfy[i] -= fulfilled_from_bin

        # --- Calculate Lost Sales ---
        lost_sales_units = demand_to_satisfy
        lost_sales_cost_step = np.sum(lost_sales_units * self.lost_sales_costs)

        # --- Calculate Holding Costs & Handle Max Inventory Constraint ---
        inventory_level_after_sales = np.sum(self.inventory_age, axis=1)
        disposed_units = np.zeros(self.n_items, dtype=int)
        # disposal_value_lost = np.zeros(self.n_items, dtype=float) # Optional: Track value lost to disposal

        for i in range(self.n_items):
            current_total_inv = inventory_level_after_sales[i]
            max_level = self.max_inventory_level[i]
            inventory_for_holding_cost = current_total_inv # Start assuming no clipping

            if current_total_inv > max_level:
                 excess = current_total_inv - max_level
                 inventory_for_holding_cost = max_level # Holding cost based on capped level
                 disposed_count_item = 0

                 # Dispose of excess from oldest bins first, updating quantity AND value
                 for age_idx_dispose in range(self.max_age - 1, -1, -1):
                     if excess <= 0: break
                     qty_in_bin = self.inventory_age[i, age_idx_dispose]
                     if qty_in_bin <= 0: continue

                     dispose_from_bin = min(excess, qty_in_bin)
                     if dispose_from_bin > 0:
                          value_in_bin = self.inventory_value[i, age_idx_dispose]
                          avg_cost_in_bin = value_in_bin / qty_in_bin if qty_in_bin > 0 else 0

                          # Update quantity and value
                          self.inventory_age[i, age_idx_dispose] -= dispose_from_bin
                          self.inventory_value[i, age_idx_dispose] -= dispose_from_bin * avg_cost_in_bin
                          self.inventory_value[i, age_idx_dispose] = max(0, self.inventory_value[i, age_idx_dispose])

                          excess -= dispose_from_bin
                          disposed_count_item += dispose_from_bin
                          # disposal_value_lost[i] += dispose_from_bin * avg_cost_in_bin # Optional track if we want to penalize having more then the maximum inventory level.

                 disposed_units[i] = disposed_count_item

            # Calculate holding cost for this item based on inventory level *after* clipping
            holding_cost_step += inventory_for_holding_cost * self.holding_costs[i]

        # Update the main inventory level state AFTER potential disposal
        self.inventory_level = np.sum(self.inventory_age, axis=1)

        # Optional: Add disposal cost (could use disposal_value_lost)
        # disposal_cost_step = np.sum(disposal_value_lost)
        # Add to total costs if needed

        return holding_cost_step, lost_sales_cost_step

    def _age_inventory_and_calc_wastage(self):
        inventory_at_start_of_aging = self.inventory_age.copy()
        value_at_start_of_aging = self.inventory_value.copy() # Copy value too
        new_inventory_age = np.zeros_like(self.inventory_age)
        new_inventory_value = np.zeros_like(self.inventory_value) # For aged value
        wastage_units_step = np.zeros(self.n_items, dtype=int)
        total_wastage_cost_step = 0.0 # Accumulates the value of wasted items

        for i in range(self.n_items):
            for age_idx in range(self.max_age):
                n_items_in_bin = inventory_at_start_of_aging[i, age_idx]
                if n_items_in_bin == 0: continue

                value_in_bin = value_at_start_of_aging[i, age_idx]
                avg_cost_in_bin = value_in_bin / n_items_in_bin if n_items_in_bin > 0 else 0

                # --- Calculate Wastage Probability ---
                cdf_age_plus_1 = self.shelf_life_cdf[i, age_idx + 1] if (age_idx + 1) < self.max_age else 1.0
                cdf_age = self.shelf_life_cdf[i, age_idx]
                prob_survival_up_to_age = 1.0 - cdf_age
                if prob_survival_up_to_age <= 1e-9:
                    prob_expire_in_next_step = 1.0
                else:
                    prob_expire_in_next_step = np.clip( (cdf_age_plus_1 - cdf_age) / prob_survival_up_to_age, 0.0, 1.0)

                # --- Simulate Binomial Wastage ---
                wasted_count = 0
                if n_items_in_bin > 0: # Here if we have to many items in the bin (should not happen but just in case)
                     if n_items_in_bin > self.max_items_for_wastage_draws:
                         wasted_count = self.env_rng.binomial(n=n_items_in_bin, p=prob_expire_in_next_step)
                     else: # This is the expected flow: we draw from a wastage set to have a repruductable result 
                         uniform_draws_for_bin = self.wastage_uniform_draws[
                             self.current_step, i, age_idx, :n_items_in_bin]
                         wasted_count = np.sum(uniform_draws_for_bin <= prob_expire_in_next_step)

                wastage_units_step[i] += wasted_count
                survivors_count = n_items_in_bin - wasted_count

                # --- Calculate Value of Wasted and Survivors ---
                value_of_wasted = wasted_count * avg_cost_in_bin
                value_of_survivors = survivors_count * avg_cost_in_bin

                total_wastage_cost_step += value_of_wasted # Accumulate cost based on value

                # --- Place Survivors (Quantity and Value) in Next Age Bin ---
                if age_idx < self.max_age - 1:
                    if survivors_count > 0:
                         # Use += in case items from different sources age into the same bin (not strictly necessary here but safer)
                         new_inventory_age[i, age_idx + 1] += survivors_count
                         new_inventory_value[i, age_idx + 1] += value_of_survivors

        # Update inventory state
        self.inventory_age = new_inventory_age
        self.inventory_value = new_inventory_value
        self.inventory_value = np.maximum(0, self.inventory_value) # Ensure non-negative value
        self.inventory_level = np.sum(self.inventory_age, axis=1)
        self.wastage = wastage_units_step

        return total_wastage_cost_step # Return the calculated value-based wastage cost

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # --- Determine Seed ---
        seed_to_use = None
        if seed is not None:
            seed_to_use = seed
            self._current_seed_state = seed
        else:
            if self._current_seed_state is None:
                seed_to_use = np.random.randint(0, 1e9)
                self._current_seed_state = seed_to_use
            else:
                seed_to_use = self._current_seed_state + 1
                self._current_seed_state += 1
        # --- Re-seed RNGs ---
        self.env_rng = default_rng(seed_to_use)
        self.stoch_model.rng = default_rng(seed_to_use)
        # --- Reset State Variables ---
        self.current_step = 0
        self.inventory_age = np.array(self.settings['initial_inventory_age'], dtype=int).copy()
        # Reset value based on how it was determined in __init__
        self.inventory_value = self.initial_inventory_value.copy()
        self.inventory_level = np.sum(self.inventory_age, axis=1)
        self.order_history = []
        self.wastage = np.zeros(self.n_items, dtype=int)
        self.last_step_costs = {}
        self.demand = np.zeros(self.n_items, dtype=int)
        self.last_step_arrivals = np.zeros(self.n_items, dtype=int)
        self.last_step_demand = np.zeros(self.n_items, dtype=int)
        # --- Generate Scenario & Pre-generate Randomness ---
        self.generate_scenario_realization()
        self._pregenerate_randomness()

        observation = self._get_observation()
        info = {
            'inventory_level': self.inventory_level.copy(),
            'inventory_age': self.inventory_age.copy(),
            'inventory_value': self.inventory_value.copy(), # Include initial value
            'wastage_units': self.wastage.copy(),
            'arrivals_units': self.last_step_arrivals.copy(),
            'demand_units': self.last_step_demand.copy()
        }
        return observation, info

    def step(self, action, verbose=False):
        if self.current_step >= self.T:
            raise Exception(f"Episode finished. Call reset(). Current step {self.current_step}, Horizon {self.T}")

        # --- START OF DAY t ---
        arrivals_today = self._receive_arrivals() # Updates age and value
        if verbose: print(f"\t[T={self.current_step}] Inv Age (Post-Arrival): \n{self.inventory_age}")
        if verbose: print(f"\t[T={self.current_step}] Inv Value (Post-Arrival): \n{self.inventory_value}")

        # --- Agent Action ---
        purchase_cost, fixed_cost = self._place_new_orders(action) # Adds purchase cost to total

        # --- DURING DAY t ---
        self.demand = self.scenario_demand[:, self.current_step]
        holding_cost, lost_sales_cost = self._satisfy_demand_and_calc_costs() # Updates age and value
        if verbose: print(f"\t[T={self.current_step}] Inv Age (Post-Sales/Clip): \n{self.inventory_age}")
        if verbose: print(f"\t[T={self.current_step}] Inv Value (Post-Sales/Clip): \n{self.inventory_value}")

        # --- END OF DAY t ---
        wastage_cost = self._age_inventory_and_calc_wastage() # Updates age and value, returns cost
        if verbose: print(f"\t[T={self.current_step}] Inv Age (Final): \n{self.inventory_age}")
        if verbose: print(f"\t[T={self.current_step}] Inv Value (Final): \n{self.inventory_value}")
        if verbose: print(f"\t[T={self.current_step}] Wastage Units: {self.wastage}, Wastage Cost (Value Based): {wastage_cost:.2f}")

        # --- Consolidate Costs & Reward ---
        self.last_step_costs = {
            'purchase_costs': purchase_cost, # Cost of orders placed *this* step
            'fixed_order_costs': fixed_cost,
            'holding_costs': holding_cost,
            'lost_sales_costs': lost_sales_cost,
            'wastage_costs': wastage_cost # Value of items wasted *this* step
        }
        reward = -sum(self.last_step_costs.values())

        # --- Prepare for Next Step (t+1) ---
        self.last_step_arrivals = arrivals_today.copy()
        self.last_step_demand = self.demand.copy()

        self.current_step += 1
        terminated = (self.current_step == self.T)
        truncated = False

        observation = self._get_observation()
        info = self.last_step_costs.copy()
        # State at END of step t (start of t+1)
        info['inventory_level'] = self.inventory_level.copy()
        info['inventory_age'] = self.inventory_age.copy()
        info['inventory_value'] = self.inventory_value.copy() # Current inventory value
        # Events during step t
        info['wastage_units'] = self.wastage.copy()
        info['arrivals_units'] = self.last_step_arrivals.copy()
        info['demand_units'] = self.last_step_demand.copy()

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only 'human' render mode is supported.")

        print("-" * 35)
        if self.current_step == 0:
             print(f"--- Initial State (Start of Step t=0) ---")
             print(f"  Inventory Age:\n{self.inventory_age}")
             print(f"  Inventory Value:\n{self.inventory_value}")
             print(f"  Inventory Level: {self.inventory_level}")
             print(f"  Outstanding Orders: {self.order_history}")
        else:
             prev_step = self.current_step - 1
             print(f"--- Details from Completed Step t={prev_step} ---")
             print(f"  Demand Encountered : {self.last_step_demand}")
             print(f"  Arrivals Received  : {self.last_step_arrivals}")
             print(f"  Wastage Units      : {self.wastage}")
             print(f"  Costs Incurred:")
             for k, v in self.last_step_costs.items():
                 # Format wastage cost specifically if needed
                 if k == 'wastage_costs':
                     print(f"    {k}: {v:.2f} (Based on actual purchase value)")
                 else:
                     print(f"    {k}: {v:.2f}")
             print(f"--- State at Start of Current Step t={self.current_step} ---")
             print(f"  Inventory Age:\n{self.inventory_age}")
             print(f"  Inventory Value:\n{self.inventory_value}") # Show value state
             print(f"  Inventory Level: {self.inventory_level}")
             print(f"  Outstanding Orders: {self.order_history}")
        print("-" * 35)

    def close(self):
        pass