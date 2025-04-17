import numpy as np
from numpy.random import Generator, default_rng # Import Generator and default_rng
import math # Needed for tiling calculation

class StochasticDemandModel():
    def __init__(self, settings, seed=None): # Add seed parameter
        """
        Initializes the StochasticDemandModel.

        Args:
            settings (dict): Configuration dictionary for the demand model.
                             Must contain 'n_items' and 'demand_distribution'.
                             'demand_distribution' must contain 'name' and relevant parameters.
            seed (int, optional): Seed for the random number generator for reproducibility.
                                  Defaults to None (non-reproducible).
        """
        self.settings = settings
        self.n_items = settings['n_items']
        self.name_distribution = settings['demand_distribution']['name']

        # --- Create a dedicated RNG instance for this object ---
        self.rng = default_rng(seed)
        # ------------------------------------------------------

        # --- Validate settings based on distribution type ---
        if self.name_distribution == 'normal':
            if not all(k in settings['demand_distribution'] for k in ['mu', 'sigma', 'seasonal_factor']):
                 raise ValueError("'mu', 'sigma', and 'seasonal_factor' must be specified for normal distribution")
            self.mu = np.array(settings['demand_distribution']['mu'])
            self.sigma = np.array(settings['demand_distribution']['sigma'])
            self.seasonal_factor = np.array(settings['demand_distribution']['seasonal_factor'])
            # self.time_horizon now represents the length of the *pattern*
            self.pattern_length = len(self.seasonal_factor)
            if self.pattern_length == 0:
                raise ValueError("seasonal_factor cannot be empty for normal distribution")
            if self.mu.shape != (self.n_items, self.pattern_length):
                raise ValueError(f"Expected mu shape ({self.n_items}, {self.pattern_length}), got {self.mu.shape}")
            if self.sigma.shape != (self.n_items, self.pattern_length):
                 raise ValueError(f"Expected sigma shape ({self.n_items}, {self.pattern_length}), got {self.sigma.shape}")

        elif self.name_distribution == 'discrete_uniform':
            # Check if low and high are present
            if 'low' not in settings['demand_distribution'] or 'high' not in settings['demand_distribution']:
                 raise ValueError("'low' and 'high' must be specified for discrete_uniform")
            self.low = settings['demand_distribution']['low']
            self.high = settings['demand_distribution']['high'] # rng.integers high bound is exclusive

        elif self.name_distribution == 'binomial':
             # Check if n and p are present
            if 'n' not in settings['demand_distribution'] or 'p' not in settings['demand_distribution']:
                 raise ValueError("'n' and 'p' must be specified for binomial")
            self.n_binom = settings['demand_distribution']['n']
            self.p_binom = settings['demand_distribution']['p']

        elif self.name_distribution == 'probability_mass_function':
             # Check if vals and probs are present
            if 'vals' not in settings['demand_distribution'] or 'probs' not in settings['demand_distribution']:
                 raise ValueError("'vals' and 'probs' must be specified for probability_mass_function")
            self.pmf_vals = settings['demand_distribution']['vals']
            self.pmf_probs = settings['demand_distribution']['probs']
            if not np.isclose(sum(self.pmf_probs), 1.0): # Use isclose for float comparison
                raise ValueError('Sum of probabilities must be 1 for probability_mass_function')
        else:
            raise ValueError(f"Unsupported distribution: {self.name_distribution}")
        # --- End Validation ---

    def generate_scenario(self, n_time_steps=1):
        """
        Generates a demand scenario for the specified number of time steps.
        If n_time_steps is larger than the pattern length for normal distribution,
        the pattern (seasonal factors, mu, sigma) is repeated.

        Args:
            n_time_steps (int, optional): The number of time steps to generate demand for.
                                           Defaults to 1.

        Returns:
            np.ndarray: A NumPy array of shape (n_items, n_time_steps) containing the generated demand.
                       Demand values are integers.
        """
        if n_time_steps <= 0:
             raise ValueError("n_time_steps must be positive.")
        output_shape = (self.n_items, n_time_steps)

        if self.name_distribution == 'normal':
            # --- Handle repeating pattern for longer horizons ---
            num_repeats = math.ceil(n_time_steps / self.pattern_length)

            # Tile seasonal factor (1D) and take the required length
            tiled_seasonal = np.tile(self.seasonal_factor, num_repeats)[:n_time_steps]
            season_t = tiled_seasonal[np.newaxis, :] # Add axis for broadcasting

            # Tile mu and sigma (2D) horizontally and take the required length
            tiled_mu = np.tile(self.mu, (1, num_repeats))[:, :n_time_steps]
            tiled_sigma = np.tile(self.sigma, (1, num_repeats))[:, :n_time_steps]
            mu_it = tiled_mu
            sigma_it = tiled_sigma
            # ----------------------------------------------------

            # Calculate means considering seasonality
            means = mu_it * season_t

            # --- Generate normal floats using the instance's RNG ---
            demand_floats = self.rng.normal(loc=means, scale=sigma_it)

            # --- Ensure non-negativity and convert to integer ---
            demand_non_negative = np.maximum(0, demand_floats) # Set negative values to 0
            demand_integers = np.round(demand_non_negative).astype(int) # Round and cast to int
            # --------------------------------------------------------
            return demand_integers

        elif self.name_distribution == 'discrete_uniform':
            # --- Use the instance's RNG ---
            # Note: rng.integers high bound is *exclusive*
            return self.rng.integers(
                low=self.low,
                high=self.high,
                size=output_shape
            )

        elif self.name_distribution == 'binomial':
             # --- Use the instance's RNG ---
            return self.rng.binomial(
                n=self.n_binom,
                p=self.p_binom,
                size=output_shape
            )

        elif self.name_distribution == 'probability_mass_function':
             # --- Use the instance's RNG and choice method ---
             # np.random.choice can directly handle multi-dimensional size
             return self.rng.choice(
                a=self.pmf_vals,         # population
                p=self.pmf_probs,        # weights
                size=output_shape,       # desired output shape
                replace=True             # Sample with replacement (like random.choices)
            )
        else:
             # This case should not be reached due to __init__ validation
             # but is good practice to include
             raise ValueError(f"Internal error: Unsupported distribution {self.name_distribution}")