import numpy as np
from numpy.random import Generator, default_rng
import gymnasium as gym
from gymnasium import spaces
from src.scenarioManager.stochasticDemandModel import StochasticDemandModel # Example import

class PerishableInvEnv(gym.Env):
    # (Keep __init__ mostly as before, ensure all relevant attributes are numpy arrays)
    def __init__(self, settings: dict, stochastic_model_settings: dict, seed=None): # Pass demand settings separately
        super(PerishableInvEnv, self).__init__()
        # --- Store the initial seed and initialize seed counter ---
        self._initial_seed = seed
        # _current_seed_state tracks the seed used for the *last* reset/initialization
        # Will be updated in the first reset call if initial_seed is None
        self._current_seed_state = self._initial_seed
        # ---------------------------------------------------------
        self.settings = settings
        # Create the demand model internally for generating scenarios
        self.stoch_model = stochastic_model_settings

        # --- Create a SEPARATE RNG for environment stochasticity ---
        self.env_rng = default_rng(seed)
        # ---------------------------------------------------------

        # --- Basic cardinalities & Parameters (convert to np arrays) ---
        self.T = settings['time_horizon']
        self.n_items = settings['n_items']
        self.n_suppliers = settings['n_suppliers']
        self.max_age = settings['max_age']
        self.max_inventory_level = np.array(settings['max_inventory_level'], dtype=int)
        self.item_supplier_matrix = np.array(settings['item_supplier_matrix'], dtype=int)
        self.unit_purchase_costs = np.array(settings['unit_purchase_costs'], dtype=float)
        self.fixed_order_costs = np.array(settings['fixed_order_costs'], dtype=float)
        self.lead_times = np.array(settings['lead_times'], dtype=int)
        # Fulfillment parameters
        self.prob_full_fulfillment = np.array(settings['prob_full_fulfillment'], dtype=float)
        self.partial_fulfillment_beta_alpha = np.array(settings['partial_fulfillment_beta_alpha'], dtype=float)
        self.partial_fulfillment_beta_beta = np.array(settings['partial_fulfillment_beta_beta'], dtype=float)
        # Shelf life
        self.shelf_life_cdf = np.array(settings['shelf_life_cdf'], dtype=float)
        # Costs
        self.holding_costs = np.array(settings['holding_costs'], dtype=float)
        self.lost_sales_costs = np.array(settings['lost_sales_costs'], dtype=float)
        # Initial state config
        self.initial_inventory_age = np.array(settings['initial_inventory_age'], dtype=int)
        # Wastage pre-generation limit
        # Set a reasonable upper bound on items needed for binomial simulation
        self.max_items_for_wastage_draws = settings.get('max_items_for_wastage_draws',
                                                      int(np.max(self.max_inventory_level) * 1.5) + 50) # Example heuristic


        # --- Validate shapes (as before) ---
        # ... ensure all validations are present ...
        assert self.prob_full_fulfillment.shape == (self.n_items, self.n_suppliers), "Shape mismatch: prob_full_fulfillment"
        assert self.partial_fulfillment_beta_alpha.shape == (self.n_items, self.n_suppliers), "Shape mismatch: partial_fulfillment_beta_alpha"
        assert self.partial_fulfillment_beta_beta.shape == (self.n_items, self.n_suppliers), "Shape mismatch: partial_fulfillment_beta_beta"


        # --- State variables & Pre-generated randomness arrays (initialized in reset) ---
        self.current_step = 0
        self.inventory_age = np.zeros((self.n_items, self.max_age), dtype=int)
        self.inventory_level = np.zeros(self.n_items, dtype=int)
        self.order_history = []
        self.demand = np.zeros(self.n_items, dtype=int)
        self.last_step_costs = {}
        self.scenario_demand = np.zeros((self.n_items, self.T), dtype=int)
        # Fulfillment randomness
        self.fulfillment_uniform_draws = np.zeros((self.T, self.n_items, self.n_suppliers))
        self.fulfillment_beta_fractions = np.zeros((self.T, self.n_items, self.n_suppliers))
        # Wastage randomness (Uniform draws for Binomial simulation)
        self.wastage_uniform_draws = np.zeros((self.T, self.n_items, self.max_age, self.max_items_for_wastage_draws))

        # --- Gym spaces (as before) ---
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(self.n_items, self.n_suppliers), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'inventory_age': spaces.Box(low=0, high=np.inf, shape=(self.n_items, self.max_age), dtype=int),
        })
        self.M = settings.get('big_M', 10000)


    def generate_scenario_realization(self):
        """Generates the demand scenario for the entire horizon T."""
        self.scenario_demand = self.stoch_model.generate_scenario(n_time_steps=self.T).astype(int)

    def _pregenerate_randomness(self):
        """Generates all random numbers needed for the episode using the environment's RNG."""
        # Fulfillment
        self.fulfillment_uniform_draws = self.env_rng.random(size=(self.T, self.n_items, self.n_suppliers))
        # Pre-generate beta fractions - might be slightly inefficient if not always needed
        # but ensures consistency.
        for i in range(self.n_items):
            for s in range(self.n_suppliers):
                 self.fulfillment_beta_fractions[:, i, s] = self.env_rng.beta(
                     a=self.partial_fulfillment_beta_alpha[i, s],
                     b=self.partial_fulfillment_beta_beta[i, s],
                     size=self.T
                 )
        # Wastage - Uniform draws
        self.wastage_uniform_draws = self.env_rng.random(
            size=(self.T, self.n_items, self.max_age, self.max_items_for_wastage_draws)
        )

    def _get_observation(self):
        """Constructs the observation dictionary for the agent."""
        # Observation should represent the state *before* the agent acts
        obs = {'inventory_age': self.inventory_age.copy()}
        # Add outstanding orders if needed by the agent's policy
        # obs['outstanding_orders'] = self._get_outstanding_orders_state()
        return obs

    def _receive_arrivals(self):
        """Applies arriving orders using pre-generated randomness."""
        arrivals_today = np.zeros(self.n_items, dtype=int)
        remaining_order_history = []

        for t_placed, i, s, qty_ordered in self.order_history:
            if t_placed + self.lead_times[i, s] == self.current_step:
                if qty_ordered > 0:
                    # --- Use Pre-generated Fulfillment Randomness ---
                    # Use t_placed as the index for the pre-generated draw
                    prob_full = self.prob_full_fulfillment[i, s]
                    uniform_draw = self.fulfillment_uniform_draws[t_placed, i, s]

                    if uniform_draw <= prob_full:
                        fulfilled_qty = qty_ordered
                    else:
                        # Use pre-generated beta fraction
                        fulfillment_fraction = self.fulfillment_beta_fractions[t_placed, i, s]
                        fulfilled_qty_float = qty_ordered * fulfillment_fraction
                        fulfilled_qty = np.round(fulfilled_qty_float).astype(int)
                    # ---------------------------------------------
                else:
                    fulfilled_qty = 0
                arrivals_today[i] += fulfilled_qty
            else:
                remaining_order_history.append((t_placed, i, s, qty_ordered))

        self.order_history = remaining_order_history
        if np.any(arrivals_today > 0):
             self.inventory_age[:, 0] += arrivals_today
             self.inventory_level = np.sum(self.inventory_age, axis=1)
        return arrivals_today

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
        """ Ages inventory, calculates wastage stochastically using pre-generated uniform draws."""
        inventory_at_start_of_aging = self.inventory_age.copy()
        new_inventory_age = np.zeros_like(self.inventory_age)
        wastage_units_step = np.zeros(self.n_items, dtype=int) # Integer wastage

        for i in range(self.n_items):
            for age_idx in range(self.max_age):
                n_items_in_bin = inventory_at_start_of_aging[i, age_idx]
                if n_items_in_bin == 0: continue

                # Probability of expiring *at* age age_idx + 1
                cdf_a = self.shelf_life_cdf[i, age_idx]
                cdf_a_minus_1 = self.shelf_life_cdf[i, age_idx - 1] if age_idx > 0 else 0.0
                prob_expire_at_this_age = np.clip(cdf_a - cdf_a_minus_1, 0.0, 1.0)

                # --- Simulate Binomial using Pre-generated Uniform Draws ---
                if n_items_in_bin > self.max_items_for_wastage_draws:
                     print(f"Warning: Item {i}, Age {age_idx+1} has {n_items_in_bin} units, exceeding pre-generation limit {self.max_items_for_wastage_draws}. Clipping.")
                     n_items_to_check = self.max_items_for_wastage_draws
                else:
                     n_items_to_check = n_items_in_bin

                # Use pre-generated U(0,1) draws for this step, item, age
                # Indices: current_step, item i, age_idx, up to n_items_to_check
                uniform_draws_for_bin = self.wastage_uniform_draws[
                    self.current_step, i, age_idx, :n_items_to_check
                ]
                wasted_count = np.sum(uniform_draws_for_bin <= prob_expire_at_this_age)
                wastage_units_step[i] += wasted_count
                # ---------------------------------------------------------

                survivors_count = n_items_in_bin - wasted_count

                if age_idx < self.max_age - 1:
                    new_inventory_age[i, age_idx + 1] = survivors_count

        self.inventory_age = new_inventory_age
        self.inventory_level = np.sum(self.inventory_age, axis=1)
        self.wastage = wastage_units_step

        wastage_cost_step = np.sum(self.wastage * self.lost_sales_costs) # Use appropriate unit cost
        return wastage_cost_step

    def reset(self, seed=None, options=None):
        """Resets the environment and pre-generates all randomness for the episode."""
        # Call Gym's reset first (handles its own internal seeding logic if applicable)
        super().reset(seed=seed)

        # --- Determine the actual seed to use for this episode ---
        seed_to_use_for_this_run = None
        if seed is not None:
            # If a seed is provided to reset, use it directly
            seed_to_use_for_this_run = seed
            self._current_seed_state = seed # Update the state for the *next* unseeded reset
        else:
            # If no seed provided to reset, use the next in sequence
            if self._current_seed_state is None:
                # This is the first reset ever and no initial seed was given
                seed_to_use_for_this_run = 0
                self._current_seed_state = 0 # Start sequence from 0
            else:
                # Increment from the last used seed
                seed_to_use_for_this_run = self._current_seed_state + 1
                self._current_seed_state += 1 # Update for the next unseeded reset

        # print(f"DEBUG: Resetting env with seed: {seed_to_use_for_this_run}") # Optional debug print

        # --- Re-seed the environment's and demand model's RNGs ---
        # Use the determined seed_to_use_for_this_run for both
        self.env_rng = default_rng(seed_to_use_for_this_run)
        self.stoch_model.rng = default_rng(seed_to_use_for_this_run) # Ensure demand model is re-seeded too
        # ---------------------------------------------------------

        # --- Reset state variables ---
        self.current_step = 0
        self.inventory_age = np.array(self.settings['initial_inventory_age'], dtype=int).copy()
        self.inventory_level = np.sum(self.inventory_age, axis=1)
        self.order_history = []
        self.wastage = np.zeros(self.n_items, dtype=int)
        self.last_step_costs = {}
        self.demand = np.zeros(self.n_items, dtype=int) # Demand for step 0 (not used before first action)
        # ---------------------------

        # --- Generate scenario and pre-generate randomness for the episode ---
        self.generate_scenario_realization()
        self._pregenerate_randomness()
        # -----------------------------------------------------------------

        observation = self._get_observation()
        info = {} # Standard Gym practice: return empty info dict on reset

        return observation, info

    def step(self, action, verbose=False):
        """Executes one time step (day t) within the environment."""
        if self.current_step >= self.T:
            raise Exception(f"Episode already finished. Current step {self.current_step}, Horizon {self.T}")

        # --- Start of Day t ---
        # 1. Receive arrivals destined for day t (Uses pre-generated randomness)
        arrivals_today = self._receive_arrivals() # Updates self.inventory_age[:, 0] and self.inventory_level

        # (State is now ready for decision making at start of day t)

        # 2. Place new orders for future delivery based on action
        purchase_cost, fixed_cost = self._place_new_orders(action) # Adds to self.order_history

        # --- During Day t ---
        # 3. Realize demand for day t (from pre-generated scenario)
        self.demand = self.scenario_demand[:, self.current_step]
        if verbose: print(f"\tTime {self.current_step}, Demand: {self.demand}")

        # 4. Satisfy demand (FIFO) and calculate related costs
        # This updates self.inventory_age and self.inventory_level based on sales/clipping
        holding_cost, lost_sales_cost = self._satisfy_demand_and_calc_costs()

        # --- End of Day t ---
        # 5. Age inventory and calculate wastage cost (Uses pre-generated randomness via simulation)
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
        info['inventory_age'] = self.inventory_age.copy()
        info['wastage_units'] = self.wastage.copy() # Now integer
        info['arrivals_units'] = arrivals_today.copy()
        info['demand_units'] = self.demand.copy()

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Renders the environment state (simple printout)."""
        # Render shows the state *after* the step completes, ready for the next decision
        print(f'--- State at Start of Time Step: {self.current_step} ---')
        print(f'  Inventory Age:\n{self.inventory_age}')
        print(f'  Inventory Level: {self.inventory_level}')
        print(f'  Demand: {self.demand}')
        print(f'  Order History: {self.order_history}')
        if self.current_step > 0: # Costs are available after step 0 completes
            print(f'  Costs from Previous Step (t={self.current_step - 1}):')
            for k, v in self.last_step_costs.items():
                print(f'    {k}: {v:.2f}')
            print(f'  Wastage Units (from end of t={self.current_step - 1}): {self.wastage}') # Wastage from previous step
        print("-" * 20)

    def close(self):
        """Perform any necessary cleanup."""
        pass