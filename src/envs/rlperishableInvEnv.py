from envs import PerishableInvEnv
import numpy as np
import gym
import copy

class PerishableInvEnvSB(PerishableInvEnv):
    def __init__(self, settings, stoch_model):
        super().__init__(settings, stoch_model)
        try:
            self.dict_obs = settings['dict_obs']
        except:
            self.dict_obs = False
        self.last_inventory = copy.copy(self.inventory_level)
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_items + 1] * self.n_machines
        )
        
        if self.dict_obs:
            self.observation_space = gym.spaces.Dict({
                'inventory_level': gym.spaces.Box(low=np.zeros(self.n_items), high=np.ones(self.n_items) * (settings['max_inventory_level'][0] + 1) * self.n_items),
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=np.zeros(self.n_items),  # high for the inventory level
                high=np.concatenate(
                    [
                        np.array(self.max_inventory_level)
                    ]),
                dtype=np.int32
            )

    def step(self, action):
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
        self.last_inventory = copy.copy(self.inventory_level)
            
        self.total_cost = self._take_action(action, self.machine_setup, self.inventory_level, self.demand)
        
        reward = -sum([ele for key, ele in self.total_cost.items()])
        
        self.current_step += 1
        done = self.current_step == self.T
        obs = self._next_observation()

        return obs, reward, done, self.total_cost

    def _next_observation(self):
        """
        Returns the next demand
        """
        obs = PerishableInvEnv._next_observation(self)
        # obs['last_inventory_level'] = copy.copy(self.last_inventory)
        if isinstance(obs, dict):    
            if not self.dict_obs:
                obs = np.concatenate(
                    (
                        obs['inventory_level'],  # n_items size
                        # obs['machine_setup'],  # n_machine size
                        # obs['last_inventory_level']  # n_items size
                    )
                )
        else:
            if self.dict_obs:
                raise('Change dict_obs to False')
        return obs
