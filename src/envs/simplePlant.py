import numpy as np
from numpy.random import Generator, default_rng
import gymnasium as gym
from gymnasium import spaces
from src.scenarioManager.stochasticDemandModel import StochasticDemandModel # Example import

class SimplePlant(gym.Env):
    # (Keep __init__ mostly as before, ensure all relevant attributes are numpy arrays)
    def __init__(self, settings: dict, stoch_model: StochasticDemandModel, seed=None):
        super(SimplePlant, self).__init__()
        self.settings = settings
        self.stoch_model = stoch_model

        self.T = settings['time_horizon']
        self.n_items = settings['n_items']
        self.n_suppliers = settings['n_suppliers']
        self.max_age = settings['max_age']

        # Convert to NumPy arrays
        self.max_inventory_level = np.array(settings['max_inventory_level'], dtype=int)
        self.item_supplier_matrix = np.array(settings['item_supplier_matrix'], dtype=int)
        self.unit_purchase_costs = np.array(settings['unit_purchase_costs'], dtype=float)
        self.fixed_order_costs = np.array(settings['fixed_order_costs'], dtype=float)
        self.lead_times = np.array(settings['lead_times'], dtype=int)
        self.fulfillment_rates = np.array(settings['fulfillment_rates'], dtype=float)
        self.shelf_life_cdf = np.array(settings['shelf_life_cdf'], dtype=float)
        self.holding_costs = np.array(settings['holding_costs'], dtype=float)
        self.lost_sales_costs = np.array(settings['lost_sales_costs'], dtype=float)
        self.initial_inventory_age = np.array(settings['initial_inventory_age'], dtype=int)

        # Validations (keep as before)
        assert self.max_inventory_level.shape == (self.n_items,), "Shape mismatch: max_inventory_level"
        # ... (include all other assertions from previous version) ...
        assert self.initial_inventory_age.shape == (self.n_items, self.max_age), "Shape mismatch: initial_inventory_age"
        assert np.all(self.shelf_life_cdf >= 0) and np.all(self.shelf_life_cdf <= 1), "CDF values must be between 0 and 1"
        for i in range(self.n_items):
             assert np.all(np.diff(self.shelf_life_cdf[i, :]) >= -1e-9), f"Shelf life CDF for item {i} is not non-decreasing"
             assert np.isclose(self.shelf_life_cdf[i, -1], 1.0), f"Shelf life CDF for item {i} does not end at 1.0"

        # State variables initialized in reset
        self.current_step = 0
        self.inventory_age = np.zeros((self.n_items, self.max_age), dtype=int)
        self.inventory_level = np.zeros(self.n_items, dtype=int)
        self.order_history = []
        self.demand = np.zeros(self.n_items, dtype=int)
        self.scenario_demand = np.zeros((self.n_items, self.T), dtype=int) # Needs T steps
        self.last_step_costs = {} # Store costs per step for info dict

        self.action_space = spaces.Box(low=0, high=np.inf, shape=(self.n_items, self.n_suppliers), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'inventory_age': spaces.Box(low=0, high=np.inf, shape=(self.n_items, self.max_age), dtype=int),
            # Include outstanding orders in observation if agent needs it
            # 'outstanding_orders': spaces.Box(...)
        })
        self.M = settings.get('big_M', 10000)


    def generate_scenario_realization(self):
        """Generates the demand scenario for the entire horizon T."""
        self.scenario_demand = self.stoch_model.generate_scenario(n_time_steps=self.T).astype(int)

    def _get_observation(self):
        """Constructs the observation dictionary for the agent."""
        # Observation should represent the state *before* the agent acts
        obs = {'inventory_age': self.inventory_age.copy()}
        # Add outstanding orders if needed by the agent's policy
        # obs['outstanding_orders'] = self._get_outstanding_orders_state()
        return obs

    def _receive_arrivals(self):
        """Applies arriving orders to the inventory (age 0 bin)."""
        arrivals_today = np.zeros(self.n_items, dtype=int)
        remaining_order_history = []
        for t_placed, i, s, qty in self.order_history:
            if t_placed + self.lead_times[i, s] == self.current_step:
                # Calculate fulfilled quantity (can be float)
                fulfilled_qty_float = qty * self.fulfillment_rates[i, s]
                # Use stochastic rounding or simple rounding - simple rounding for now
                arrivals_today[i] += np.round(fulfilled_qty_float).astype(int)
            else:
                remaining_order_history.append((t_placed, i, s, qty))
        self.order_history = remaining_order_history
        self.inventory_age[:, 0] += arrivals_today # Add arrivals to the newest age bin (index 0)
        self.inventory_level = np.sum(self.inventory_age, axis=1) # Update total level
        return arrivals_today # Return for info dict

    def _place_new_orders(self, action):
        """Places new orders based on action and calculates ordering costs."""
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
                    # Append order: (time_placed, item, supplier, quantity)
                    # Time placed is the *current* step, delivery happens later
                    self.order_history.append((self.current_step, i, s, qty_ordered))

        fixed_cost_step = np.sum(self.fixed_order_costs * supplier_used_this_step)
        return purchase_cost_step, fixed_cost_step

    def _satisfy_demand_and_calc_costs(self):
        """Satisfies demand using FIFO, calculates lost sales and holding costs."""
        lost_sales_cost_step = 0.0
        holding_cost_step = 0.0
        demand_to_satisfy = self.demand.copy()

        # Satisfy demand using FIFO (oldest first)
        for i in range(self.n_items):
            for age_idx in range(self.max_age - 1, -1, -1): # Oldest (index max_age-1) to newest (index 0)
                if demand_to_satisfy[i] <= 0: break
                available = self.inventory_age[i, age_idx]
                fulfilled = min(available, demand_to_satisfy[i])
                self.inventory_age[i, age_idx] -= fulfilled # Reduce inventory
                demand_to_satisfy[i] -= fulfilled

            # Calculate lost sales cost
            if demand_to_satisfy[i] > 0:
                lost_sales_cost_step += demand_to_satisfy[i] * self.lost_sales_costs[i]

            # Calculate holding cost on inventory REMAINING AFTER sales, BEFORE aging
            # Apply max inventory constraint AFTER sales
            current_total_inv = np.sum(self.inventory_age[i, :])
            if current_total_inv > self.max_inventory_level[i]:
                 excess = current_total_inv - self.max_inventory_level[i]
                 # print(f"Warning: Item {i} inventory {current_total_inv} exceeds max {self.max_inventory_level[i]}. Disposing of {excess} units.")
                 # Dispose of excess, typically oldest first (requires another loop or careful indexing)
                 # For simplicity, we'll just cap the inventory for holding cost calculation.
                 # A more realistic model would track disposal cost here.
                 inventory_for_holding = self.max_inventory_level[i]
                 # Need to actually remove 'excess' items from self.inventory_age, oldest first
                 disposed = 0
                 for age_idx_dispose in range(self.max_age - 1, -1, -1):
                     if excess <= 0: break
                     dispose_qty = min(excess, self.inventory_age[i, age_idx_dispose])
                     self.inventory_age[i, age_idx_dispose] -= dispose_qty
                     excess -= dispose_qty
                     disposed += dispose_qty
                 # We are not explicitly adding a disposal cost here, but could.
            else:
                 inventory_for_holding = current_total_inv

            holding_cost_step += inventory_for_holding * self.holding_costs[i]

        # Update total inventory level after sales and potential disposal
        self.inventory_level = np.sum(self.inventory_age, axis=1)

        return holding_cost_step, lost_sales_cost_step

    def _age_inventory_and_calc_wastage(self):
        """
        Ages inventory remaining after sales, calculates wastage based on expiration
        during this aging step, and determines the inventory state for the START
        of the next period (before next arrivals).

        Returns:
            float: Total wastage cost for the current step.
        """
        inventory_at_start_of_aging = self.inventory_age.copy() # Inventory after sales
        new_inventory_age = np.zeros_like(self.inventory_age) # Will be state at start of t+1 BEFORE arrivals
        wastage_units_step = np.zeros(self.n_items, dtype=float)

        for i in range(self.n_items):
            # Calculate wastage first by seeing what expires *from* each age bin
            for age_idx in range(self.max_age): # Check ages 0 to max_age-1
                # Probability of expiring *at* age age_idx + 1
                # P(lifetime = age_idx + 1) = CDF(age_idx + 1) - CDF(age_idx)
                # Indices are age_idx and age_idx - 1
                cdf_a = self.shelf_life_cdf[i, age_idx]
                cdf_a_minus_1 = self.shelf_life_cdf[i, age_idx - 1] if age_idx > 0 else 0
                prob_expire_at_this_age = max(0, cdf_a - cdf_a_minus_1)

                wastage_from_this_bin = inventory_at_start_of_aging[i, age_idx] * prob_expire_at_this_age
                wastage_units_step[i] += wastage_from_this_bin

                # Calculate survivors from this bin that move to the next age
                # These are items of age 'age_idx' surviving to become age 'age_idx + 1'
                # They survive if their lifetime is > age_idx + 1
                # Prob survive = 1 - P(lifetime <= age_idx + 1) = 1 - CDF(age_idx + 1)
                # CDF index is age_idx
                if age_idx < self.max_age - 1: # Only non-oldest items can age further
                    prob_survive_to_next_age = 1.0 - cdf_a # Probability it lasts longer than current age index + 1
                    survivors_float = inventory_at_start_of_aging[i, age_idx] * prob_survive_to_next_age
                    # Assign to the *next* age bin in the new state array
                    new_inventory_age[i, age_idx + 1] = np.round(survivors_float).astype(int)
                    # Alternative: calculate survivors based on wastage
                    # survivors_from_this_bin = inventory_at_start_of_aging[i, age_idx] - wastage_from_this_bin
                    # if age_idx < self.max_age - 1:
                    #    new_inventory_age[i, age_idx + 1] = np.round(survivors_from_this_bin).astype(int)


        # Update the state for the start of the next period (before arrivals)
        self.inventory_age = new_inventory_age
        self.inventory_level = np.sum(self.inventory_age, axis=1)
        self.wastage = wastage_units_step # Store wastage that occurred *during* this step's end

        # Calculate wastage cost
        wastage_cost_step = np.sum(self.wastage * self.lost_sales_costs) # Or a specific wastage unit cost
        return wastage_cost_step

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory_age = np.array(self.settings['initial_inventory_age'], dtype=int).copy()
        self.inventory_level = np.sum(self.inventory_age, axis=1)
        self.order_history = []
        self.wastage = np.zeros(self.n_items)
        self.last_step_costs = {} # Clear last step costs
        self.generate_scenario_realization()
        # No demand occurs before the first action in step 0
        self.demand = np.zeros(self.n_items, dtype=int)

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action, verbose=False):
        """Executes one time step (day t) within the environment."""
        if self.current_step >= self.T:
            raise Exception(f"Episode already finished. Current step {self.current_step}, Horizon {self.T}")

        # --- Start of Day t ---
        # 1. Receive arrivals destined for day t
        arrivals_today = self._receive_arrivals() # Updates self.inventory_age[:, 0] and self.inventory_level

        # (State is now ready for decision making at start of day t)

        # 2. Place new orders for future delivery based on action
        purchase_cost, fixed_cost = self._place_new_orders(action) # Adds to self.order_history

        # --- During Day t ---
        # 3. Realize demand for day t
        self.demand = self.scenario_demand[:, self.current_step]
        if verbose: print(f"\tTime {self.current_step}, Demand: {self.demand}")

        # 4. Satisfy demand and calculate holding/lost sales costs
        # This updates self.inventory_age and self.inventory_level based on sales/clipping
        holding_cost, lost_sales_cost = self._satisfy_demand_and_calc_costs()

        # --- End of Day t ---
        # 5. Age inventory and calculate wastage cost
        # This updates self.inventory_age and self.inventory_level for the start of day t+1
        # It also calculates self.wastage based on inventory *after* sales
        wastage_cost = self._age_inventory_and_calc_wastage()

        # 6. Store costs and calculate reward
        self.last_step_costs = {
            'purchase_costs': purchase_cost,
            'fixed_order_costs': fixed_cost,
            'lost_sales_costs': lost_sales_cost,
            'holding_costs': holding_cost,
            'wastage_costs': wastage_cost
        }
        reward = -sum(self.last_step_costs.values())

        # 7. Prepare for next step
        self.current_step += 1
        terminated = (self.current_step == self.T)
        truncated = False
        observation = self._get_observation() # State for the start of step t+1

        # 8. Gather info
        info = self.last_step_costs.copy()
        info['inventory_level'] = self.inventory_level.copy()
        # Return inventory_age *after* aging, ready for next step's decision
        info['inventory_age'] = self.inventory_age.copy()
        info['wastage_units'] = self.wastage.copy()
        info['arrivals_units'] = arrivals_today.copy()
        info['demand_units'] = self.demand.copy() # Demand that occurred this step

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Renders the environment state (simple printout)."""
        # Render shows the state *after* the step completes, ready for the next decision
        print(f'--- State at Start of Time Step: {self.current_step} ---')
        print(f'  Inventory Age:\n{self.inventory_age}')
        print(f'  Inventory Level: {self.inventory_level}')
        if self.current_step > 0: # Costs are available after step 0 completes
            print(f'  Costs from Previous Step (t={self.current_step - 1}):')
            for k, v in self.last_step_costs.items():
                print(f'    {k}: {v:.2f}')
            print(f'  Wastage Units (from end of t={self.current_step - 1}): {self.wastage}') # Wastage from previous step
        print("-" * 20)

    def close(self):
        """Perform any necessary cleanup."""
        pass