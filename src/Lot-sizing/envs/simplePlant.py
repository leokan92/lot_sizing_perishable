# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scenarioManager import StochasticDemandModel

class SimplePlant(gym.Env):

    def __init__(self, settings: dict, stoch_model: StochasticDemandModel):
        
        super(SimplePlant, self).__init__()
        self.settings = settings
        # Basic cardinalities:
        self.T = settings['time_horizon']
        self.n_items = settings['n_items']
        self.n_machines = settings['n_machines']

        # Caracteristics:
        self.machine_production_matrix = settings['machine_production']
        # for each machine the states in which it can be:
        self.possible_machine_production = []
        for i in range(self.n_machines):
            self.possible_machine_production.append([0])
            self.possible_machine_production[-1].extend(
                list(
                    np.nonzero(settings['machine_production'][i])[0] + 1
                )
            )

        self.max_inventory_level = settings['max_inventory_level']

        # Costs:
        self.holding_costs = settings['holding_costs']
        self.lost_sales_costs = settings['lost_sales_costs']
        self.order_costs = settings['order_costs']
        self.shelf_life = settings['shelf_life']
        self.waste = np.zeros(self.n_items)

        # Initial State:
        self.current_step = 0
        self.machine_initial_setup = settings['initial_setup'] # This is the vector indicating the index of the item type for which the initial setup of the machine is prepared
        self.initial_inventory = settings['initial_inventory']
        self.inventory_level = self.initial_inventory.copy()
        self.machine_setup = self.machine_initial_setup.copy()

        # Demand generation:
        self.stoch_model = stoch_model
        # Generate demand scenario:
        self.generate_scenario_realization()

    def generate_scenario_realization(self):
        self.scenario_demand = self.stoch_model.generate_scenario(
            n_time_steps = self.T + 1
        )

    def _random_initial_state(self):
        for i in range(self.n_items):
            self.inventory_level[i] = np.random.randint(0, self.max_inventory_level[i])
        for m in range(self.n_machines):
            non_zero_prod = [i for i, e in enumerate(self.machine_production_matrix[m]) if e != 0]
            self.machine_setup[m] = np.random.choice(non_zero_prod) + 1
        # Update initial value (useful for restarting time)
        self.initial_inventory = self.inventory_level.copy()
        self.machine_initial_setup = self.machine_setup.copy()

    def _next_observation(self):
        """
        Returns the next demand
        """
        self.demand = self.scenario_demand[:, self.current_step]
        return {
            # 'demand': self.demand,
            'inventory_level': self.inventory_level,
            'machine_setup': self.machine_setup
        }

    def _post_decision(self, action, machine_setup, inventory_level, order_costs):
        waste = np.zeros(self.n_items, dtype=int)
        # if we are just changing the setup, we use the order cost matrix with the corresponding position given by the actual setup and the new setup
        for m in range(self.n_machines):   
            if action[m] != 0: # if the machine is not idle
                # 1. IF NEEDED CHANGE SETUP
                if machine_setup[m] != action[m] and action[m] != 0:
                    order_costs[m] = self.setup_costs[m][action[m] - 1] 
                    waste[m] = self.setup_loss[m][action[m] - 1]
                machine_setup[m] = action[m]
                # 2. PRODUCTION
                production = self.machine_production_matrix[m][action[m] - 1] - waste[m]
                inventory_level[action[m] - 1] += production
            else:
                machine_setup[m] = 0

    def _satisfy_demand(self, lost_sales, inventory_level, demand, holding_costs):
        # 3. SATISFYING DEMAND
        for i in range(self.n_items):
            inventory_level[i] -= demand[i]
            if inventory_level[i] < 0:
                lost_sales[i] -= inventory_level[i] * self.lost_sales_costs[i]
                inventory_level[i] = 0
            elif inventory_level[i] > self.max_inventory_level[i]:
                inventory_level[i] = self.max_inventory_level[i]
            # 4. HOLDING COSTS
            holding_costs[i] += inventory_level[i] * self.holding_costs[i]
            # 5. WASTE (SHRINKAGE)
            self.waste[i] = inventory_level[i] * (1 - np.exp(-1 / self.shelf_life[i]))
            inventory_level[i] -= self.waste[i]

    def _take_action(self, action, machine_setup, inventory_level, demand):

        order_costs = np.zeros(self.n_machines)
        lost_sales = np.zeros(self.n_items)
        holding_costs = np.zeros(self.n_items)
        # 1. POST DECISION
        self._post_decision(action, machine_setup, inventory_level, order_costs)
        # 2. RND NOISE
        self._satisfy_demand(lost_sales, inventory_level, demand, holding_costs)
        return {
            'order_costs': sum(order_costs),
            'lost_sales': sum(lost_sales),
            'holding_costs': sum(holding_costs),
        }

    def reset_time(self):
        # State variable:
        self.current_step = 0
        self.inventory_level = self.initial_inventory.copy()
        self.machine_setup = self.machine_initial_setup.copy()
        # Monitoring variables
        self.total_cost = {
            "setup_costs": 0.0,
            "lost_sales": 0.0,
            "holding_costs": 0.0,
        }
        obs = self._next_observation()
        return obs

    def reset(self): 
        """
        Reset all environment variables important for the simulation.
            - Inventory
            - Setup
            - Demand_function
            - Current_step
        """
            
        # State variable:
        self.current_step = 0
        self._random_initial_state()
   
        # Monitoring variables
        self.total_cost = {
            "setup_costs": 0.0,
            "lost_sales": 0.0,
            "holding_costs": 0.0,
        } 
        self.generate_scenario_realization()
        # ci serve avere una domanda diversa quando resetto? secondo me no.
        obs = self._next_observation()
        return obs

    def step(self, action, verbose=False):
        """
        Step method: Execute one time step within the environment

        Parameters
        ----------
        action : action given by the agent

        Returns
        -------
        obs : Observation of the state give the method _next_observation
        reward : Cost given by the _reward method
        done : returns True or False given by the _done method
        dict : possible information for control to environment monitoring

        """
        # TODO: ragionare bene sul tempo
        #if len(set(action)) != len(action):
        #    print("Error, at least two machines produce the same item")
        #    quit()

        if verbose:
            print(f"\t demand_{self.current_step}: {self.demand}")
        self.total_cost = self._take_action(
            action,
            self.machine_setup,
            self.inventory_level,
            self.demand
        )
        
        reward = sum([ele for _, ele in self.total_cost.items()])
        
        self.current_step += 1
        done = self.current_step == self.T
        obs = self._next_observation()
        info = {key: ele for key, ele in self.total_cost.items()}
        info['inventory'] = self.inventory_level
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Time: {self.current_step}')
        print(f'  inventory: {self.inventory_level}')
        print(f'  setup: {self.machine_setup}')
        print(f'  waste: {self.waste}')
        print(f'    order_costs: {self.total_cost["order_costs"]}')
        print(f'    lost_sales: {self.total_cost["lost_sales"]}')
        print(f'    holding_costs: {self.total_cost["holding_costs"]}')
        # print(f'\t demand + 1: {self.demand}')

    def plot_production_matrix(self, file_path=None):
        
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)
        plt.rc('axes', linewidth=2)
        tmp = np.array(self.machine_production_matrix).T
        fig = plt.figure(figsize=(8,12), dpi= 100, facecolor='w', edgecolor='k')
        data_masked = np.ma.masked_where(
            tmp == 0, tmp
        )
        cax = plt.imshow(data_masked, interpolation = 'none', vmin = 1)

        fig.colorbar(cax)
        plt.xlabel('Items',fontsize=18,weight='bold')
        plt.ylabel('Machines',fontsize=18,weight='bold')

        if file_path:
            fig.savefig(file_path)
        else:
            plt.show()
        plt.close()
