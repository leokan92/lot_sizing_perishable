import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError:
    print("FATAL ERROR: stable-baselines3 is not installed. Please install it using 'pip install stable-baselines3'", file=sys.stderr)
    sys.exit(1)


class _SB3PerishableEnvWrapper(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self,
                 base_env,
                 algorithm_name: str,
                 max_order_quantity: int = None,
                 dqn_num_actions: int = 256,
                 dqn_action_seed: int = 12345,
                 reward_scale: float = 1.0):
        super().__init__()
        self.base_env = base_env
        self.algorithm_name = str(algorithm_name).upper()
        self.reward_scale = float(reward_scale)
        self.n_items = int(base_env.n_items)
        self.n_suppliers = int(base_env.n_suppliers)
        self.max_age = int(base_env.max_age)
        self.T = int(base_env.T)
        self.item_supplier_matrix = np.array(base_env.item_supplier_matrix, dtype=int)
        self._action_shape = (self.n_items, self.n_suppliers)

        if max_order_quantity is None:
            max_order_quantity = int(np.max(base_env.max_inventory_level))
        self.max_order_quantity = max(1, int(max_order_quantity))

        self.obs_dim = self.n_items * self.max_age + 1
        obs_high_inventory = np.repeat(np.array(base_env.max_inventory_level, dtype=np.float32), self.max_age)
        obs_high = np.concatenate([obs_high_inventory, np.array([float(self.T)], dtype=np.float32)])
        self.observation_space = spaces.Box(
            low=np.zeros(self.obs_dim, dtype=np.float32),
            high=obs_high,
            dtype=np.float32
        )

        if self.algorithm_name == "DQN":
            self.dqn_num_actions = max(2, int(dqn_num_actions))
            self._build_dqn_action_catalog(seed=int(dqn_action_seed))
            self.action_space = spaces.Discrete(self.dqn_num_actions)
        else:
            # Only expose valid (item, supplier) positions to avoid 0-width
            # dimensions that cause NaN in SAC's action normalization.
            self._valid_positions = np.argwhere(self.item_supplier_matrix == 1)
            self._n_valid = len(self._valid_positions)
            self.action_space = spaces.Box(
                low=np.zeros(self._n_valid, dtype=np.float32),
                high=np.full(self._n_valid, float(self.max_order_quantity), dtype=np.float32),
                dtype=np.float32
            )

    def _build_dqn_action_catalog(self, seed: int):
        rng = np.random.default_rng(seed)
        catalog = np.zeros((self.dqn_num_actions, self.n_items, self.n_suppliers), dtype=np.float32)
        valid_positions = np.argwhere(self.item_supplier_matrix == 1)

        for a in range(self.dqn_num_actions):
            if len(valid_positions) == 0:
                break

            n_active = int(rng.integers(low=1, high=min(4, len(valid_positions)) + 1))
            chosen = rng.choice(len(valid_positions), size=n_active, replace=False)

            for idx in chosen:
                i, s = valid_positions[idx]
                qty = int(rng.integers(0, self.max_order_quantity + 1))
                catalog[a, i, s] = float(qty)

        self._dqn_action_catalog = catalog

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        inventory_age = np.asarray(obs_dict['inventory_age'], dtype=np.float32).reshape(-1)
        current_step = float(obs_dict['current_step'])
        return np.concatenate([inventory_age, np.array([current_step], dtype=np.float32)], axis=0)

    def _decode_action(self, action) -> np.ndarray:
        if self.algorithm_name == "DQN":
            action_idx = int(np.asarray(action).reshape(-1)[0])
            action_idx = int(np.clip(action_idx, 0, self.dqn_num_actions - 1))
            action_matrix = self._dqn_action_catalog[action_idx].copy()
        else:
            # Reconstruct full (n_items, n_suppliers) matrix from flat valid-only vector
            flat = np.asarray(action, dtype=np.float32).reshape(-1)
            action_matrix = np.zeros(self._action_shape, dtype=np.float32)
            for k, (i, s) in enumerate(self._valid_positions):
                action_matrix[i, s] = flat[k]
            action_matrix = np.floor(np.maximum(0.0, action_matrix)).astype(np.float32)
            action_matrix = np.minimum(action_matrix, float(self.max_order_quantity))

        return action_matrix

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def step(self, action):
        decoded_action = self._decode_action(action)
        obs, reward, terminated, truncated, info = self.base_env.step(decoded_action, verbose=False)
        return self._flatten_obs(obs), float(reward) * self.reward_scale, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


class StableBaselinesAgent:
    def __init__(self,
                 env,
                 algorithm: str = "PPO",
                 policy: str = "MlpPolicy",
                 total_timesteps: int = 10000,
                 num_final_eval_episodes: int = 50,
                 deterministic_eval: bool = True,
                 train_log_interval: int = 10,
                 sb3_params: dict = None,
                 max_order_quantity: int = None,
                 dqn_num_actions: int = 256,
                 dqn_action_seed: int = 12345,
                 n_envs: int = 1,
                 vec_env_type: str = "dummy",
                 load_policy_path: str = None,
                 save_policy_path: str = None,
                 device: str = "auto",
                 verbose: int = 1,
                 **kwargs):

        self.env = env
        self.algorithm = str(algorithm).upper()
        self.policy = policy
        self.total_timesteps = int(total_timesteps)
        self.num_final_eval_episodes = int(num_final_eval_episodes)
        self.deterministic_eval = bool(deterministic_eval)
        self.train_log_interval = int(train_log_interval)
        self.sb3_params = sb3_params or {}
        self.load_policy_path = load_policy_path
        self.save_policy_path = save_policy_path
        self.device = device
        self.verbose = int(verbose)
        self.n_envs = max(1, int(n_envs))
        self.vec_env_type = str(vec_env_type).lower()

        self.reward_scale = float(kwargs.get('reward_scale', 1.0))

        # Wrapper kwargs shared across all env copies
        self._wrapper_kwargs = dict(
            algorithm_name=self.algorithm,
            max_order_quantity=max_order_quantity,
            dqn_num_actions=dqn_num_actions,
            dqn_action_seed=dqn_action_seed,
            reward_scale=self.reward_scale,
        )

        algo_map = {
            "PPO": PPO,
            "SAC": SAC,
            "DQN": DQN,
        }
        if self.algorithm not in algo_map:
            raise ValueError(f"Unsupported SB3 algorithm '{self.algorithm}'. Use one of: {list(algo_map.keys())}")
        self.AlgoClass = algo_map[self.algorithm]

        # --- Build vectorized training environment ---
        base_seed = getattr(self.env, '_initial_seed', None) or 0
        self.vec_env = self._build_vec_env(base_seed)
        print(f"Created {self.vec_env_type.upper()} VecEnv with {self.n_envs} parallel env(s)")

        # --- Single (non-vectorized) wrapper for evaluation ---
        self.eval_env = _SB3PerishableEnvWrapper(
            base_env=self.env,
            **self._wrapper_kwargs,
        )

        seed = getattr(self.env, '_initial_seed', None)
        if seed is not None and 'seed' not in self.sb3_params:
            self.sb3_params['seed'] = int(seed)

        if self.load_policy_path:
            abs_load_path = os.path.abspath(self.load_policy_path)
            if not os.path.exists(abs_load_path):
                raise FileNotFoundError(f"SB3 model file not found: {abs_load_path}")

            print(f"\n--- Loading pre-trained SB3 policy ({self.algorithm}) ---")
            self.model = self.AlgoClass.load(abs_load_path, env=self.vec_env, device=self.device)
            print(f"Loaded SB3 policy from: {abs_load_path}")
        else:
            print(f"\n--- Training SB3 policy with {self.algorithm} ---")
            print(f"Total timesteps: {self.total_timesteps}  |  n_envs: {self.n_envs}")
            self.model = self.AlgoClass(
                self.policy,
                self.vec_env,
                verbose=self.verbose,
                device=self.device,
                **self.sb3_params
            )
            self.model.learn(total_timesteps=self.total_timesteps, log_interval=self.train_log_interval)

            if self.save_policy_path:
                abs_save_path = os.path.abspath(self.save_policy_path)
                os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)
                self.model.save(abs_save_path)
                print(f"Saved SB3 policy to: {abs_save_path}")

    # ------------------------------------------------------------------
    # Vectorized-environment factory
    # ------------------------------------------------------------------
    def _build_vec_env(self, base_seed: int):
        """Create a DummyVecEnv or SubprocVecEnv with *n_envs* copies."""
        env_settings = self.env.settings
        stoch_model_settings = self.env.stoch_model
        wrapper_kwargs = self._wrapper_kwargs

        def _make_env(rank: int, seed: int):
            """Return a zero-argument callable that builds one wrapped env."""
            def _init():
                from src.envs.perishableInvEnv import PerishableInvEnv
                from src.scenarioManager.stochasticDemandModel import StochasticDemandModel

                # Each copy gets its own StochasticDemandModel with a unique seed
                env_seed = seed + rank
                stoch_copy = StochasticDemandModel(env_settings, seed=env_seed)
                env_copy = PerishableInvEnv(env_settings, stoch_copy, seed=env_seed)
                return _SB3PerishableEnvWrapper(base_env=env_copy, **wrapper_kwargs)
            return _init

        env_fns = [_make_env(rank=i, seed=base_seed) for i in range(self.n_envs)]

        if self.vec_env_type == "subproc":
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def run(self, render_steps=False, verbose=False):
        all_episode_rewards = []
        print(
            f"\nRunning final evaluation with SB3 ({self.algorithm}) for "
            f"{self.num_final_eval_episodes} episode(s)..."
        )

        for episode_idx in range(self.num_final_eval_episodes):
            obs, _ = self.eval_env.reset()
            terminated = False
            truncated = False
            total_reward_episode = 0.0

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic_eval)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward_episode += float(reward)

                if render_steps:
                    self.eval_env.render()

            if verbose:
                print(f"Evaluation Episode {episode_idx + 1}: Total Reward: {total_reward_episode:.2f}")
            all_episode_rewards.append(total_reward_episode)

        if self.num_final_eval_episodes > 0:
            avg_final_reward = np.mean(all_episode_rewards)
            print(
                f"Average reward over {self.num_final_eval_episodes} final evaluation episodes: "
                f"{avg_final_reward:.2f}"
            )

        return all_episode_rewards
