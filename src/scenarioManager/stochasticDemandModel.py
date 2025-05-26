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

        self.rng = default_rng(seed)

        if self.name_distribution == 'normal' or self.name_distribution == 'normal_truncated_at_zero':
            if not all(k in settings['demand_distribution'] for k in ['mu', 'sigma']):
                 raise ValueError("'mu' and 'sigma' must be specified for normal distributions")
            
            self.mu = np.array(settings['demand_distribution']['mu'])
            self.sigma = np.array(settings['demand_distribution']['sigma'])
            
            # Handle seasonal_factor: it can be an array or null
            self.seasonal_factor_input = settings['demand_distribution'].get('seasonal_factor', None) # Get with default None

            if self.seasonal_factor_input is not None:
                self.seasonal_factor = np.array(self.seasonal_factor_input)
                self.pattern_length = len(self.seasonal_factor)
                if self.pattern_length == 0:
                    # This implies seasonal_factor was an empty list, treat as no seasonality
                    print("Warning: seasonal_factor was an empty list. Treating as no seasonality (factor=1.0).")
                    self.seasonal_factor = None 
                    # For pattern_length, use mu's time dimension if no seasonal_factor
                    if self.mu.ndim == 2:
                        self.pattern_length = self.mu.shape[1]
                    elif self.mu.ndim == 1 and self.n_items == 1: # mu is 1D for single item
                        self.pattern_length = self.mu.shape[0]
                    else:
                        raise ValueError("Cannot determine pattern_length if mu is 1D and n_items > 1 without seasonal_factor.")

                if self.seasonal_factor is not None and self.mu.shape != (self.n_items, self.pattern_length):
                     raise ValueError(f"Expected mu shape ({self.n_items}, {self.pattern_length}), got {self.mu.shape} when seasonal_factor is present.")
                if self.seasonal_factor is not None and self.sigma.shape != (self.n_items, self.pattern_length):
                     raise ValueError(f"Expected sigma shape ({self.n_items}, {self.pattern_length}), got {self.sigma.shape} when seasonal_factor is present.")
            else: # seasonal_factor_input is None
                self.seasonal_factor = None
                # If no seasonal_factor, mu and sigma define the pattern directly for their full length
                if self.mu.ndim == 2:
                    self.pattern_length = self.mu.shape[1]
                elif self.mu.ndim == 1 and self.n_items == 1: # mu is 1D for single item
                    self.pattern_length = self.mu.shape[0]
                else: # mu is 1D for multiple items (implies same mu pattern for all items, or error)
                    if self.mu.ndim == 1 and self.mu.shape[0] > 0 :
                        self.pattern_length = self.mu.shape[0]
                        # We'll need to broadcast/tile mu later if it's 1D for multiple items
                    else:
                        raise ValueError(f"Mu shape {self.mu.shape} is incompatible with n_items {self.n_items} when seasonal_factor is null.")
                
                # Validate sigma against this pattern_length
                if self.sigma.ndim == 2 and self.sigma.shape[1] != self.pattern_length:
                    raise ValueError(f"Sigma time dimension {self.sigma.shape[1]} must match mu time dimension {self.pattern_length} when seasonal_factor is null.")
                elif self.sigma.ndim == 1 and self.sigma.shape[0] != self.pattern_length and self.n_items == 1:
                    raise ValueError(f"Sigma length {self.sigma.shape[0]} must match mu length {self.pattern_length} for single item when seasonal_factor is null.")
                elif self.sigma.ndim == 1 and self.sigma.shape[0] != self.pattern_length and self.n_items > 1:
                     raise ValueError(f"Sigma length {self.sigma.shape[0]} must match mu pattern length {self.pattern_length} when seasonal_factor is null.")


            if self.pattern_length == 0: # Should be caught above, but defensive
                raise ValueError("Pattern length for mu/sigma/seasonal_factor cannot be zero.")

        elif self.name_distribution == 'poisson': # ADDED POISSON
            if 'lambda' not in settings['demand_distribution']:
                raise ValueError("'lambda' must be specified for poisson distribution")
            self.lambda_param = np.array(settings['demand_distribution']['lambda'])
            
            self.seasonal_factor_input = settings['demand_distribution'].get('seasonal_factor', None)

            if self.seasonal_factor_input is not None:
                self.seasonal_factor = np.array(self.seasonal_factor_input)
                self.pattern_length = len(self.seasonal_factor)
                if self.pattern_length == 0:
                    print("Warning: seasonal_factor was an empty list for Poisson. Treating as no seasonality.")
                    self.seasonal_factor = None
                    if self.lambda_param.ndim == 2: self.pattern_length = self.lambda_param.shape[1]
                    elif self.lambda_param.ndim == 1 and self.n_items == 1: self.pattern_length = self.lambda_param.shape[0]
                    else: raise ValueError("Cannot determine pattern_length for Poisson without seasonal_factor and ambiguous lambda.")
                if self.seasonal_factor is not None and self.lambda_param.shape != (self.n_items, self.pattern_length):
                    raise ValueError(f"Expected lambda shape ({self.n_items}, {self.pattern_length}), got {self.lambda_param.shape} for Poisson with seasonal_factor.")
            else: # seasonal_factor_input is None
                self.seasonal_factor = None
                if self.lambda_param.ndim == 2: self.pattern_length = self.lambda_param.shape[1]
                elif self.lambda_param.ndim == 1 and self.n_items == 1: self.pattern_length = self.lambda_param.shape[0]
                elif self.lambda_param.ndim == 1 and self.lambda_param.shape[0] > 0: # Lambda is 1D pattern for all items
                    self.pattern_length = self.lambda_param.shape[0]
                else: raise ValueError(f"Lambda shape {self.lambda_param.shape} incompatible for Poisson when seasonal_factor is null.")
            
            if self.pattern_length == 0:
                raise ValueError("Pattern length for lambda/seasonal_factor cannot be zero for Poisson.")


        elif self.name_distribution == 'discrete_uniform':
            if 'low' not in settings['demand_distribution'] or 'high' not in settings['demand_distribution']:
                 raise ValueError("'low' and 'high' must be specified for discrete_uniform")
            self.low = settings['demand_distribution']['low']
            self.high = settings['demand_distribution']['high']

        elif self.name_distribution == 'binomial':
            if 'n' not in settings['demand_distribution'] or 'p' not in settings['demand_distribution']:
                 raise ValueError("'n' and 'p' must be specified for binomial")
            self.n_binom = settings['demand_distribution']['n']
            self.p_binom = settings['demand_distribution']['p']

        elif self.name_distribution == 'probability_mass_function':
            if 'vals' not in settings['demand_distribution'] or 'probs' not in settings['demand_distribution']:
                 raise ValueError("'vals' and 'probs' must be specified for probability_mass_function")
            self.pmf_vals = settings['demand_distribution']['vals']
            self.pmf_probs = settings['demand_distribution']['probs']
            if not np.isclose(sum(self.pmf_probs), 1.0):
                raise ValueError('Sum of probabilities must be 1 for probability_mass_function')
        else:
            raise ValueError(f"Unsupported distribution: {self.name_distribution}")

    def generate_scenario(self, n_time_steps=1):
        if n_time_steps <= 0:
             raise ValueError("n_time_steps must be positive.")
        output_shape = (self.n_items, n_time_steps)

        if self.name_distribution == 'normal' or self.name_distribution == 'normal_truncated_at_zero':
            num_repeats = math.ceil(n_time_steps / self.pattern_length)

            # Tile mu
            if self.mu.ndim == 1 and self.n_items > 1: # mu is 1D pattern, to be applied to all n_items
                tiled_mu_pattern = np.tile(self.mu, num_repeats)[:n_time_steps]
                tiled_mu = np.tile(tiled_mu_pattern, (self.n_items, 1))
            elif self.mu.ndim == 1 and self.n_items == 1: # mu is 1D pattern for the single item
                 tiled_mu_pattern = np.tile(self.mu, num_repeats)[:n_time_steps]
                 tiled_mu = tiled_mu_pattern.reshape(1, n_time_steps) # Ensure 2D
            else: # mu is already (n_items, pattern_length)
                tiled_mu = np.tile(self.mu, (1, num_repeats))[:, :n_time_steps]

            # Tile sigma (similar logic to mu)
            if self.sigma.ndim == 1 and self.n_items > 1:
                tiled_sigma_pattern = np.tile(self.sigma, num_repeats)[:n_time_steps]
                tiled_sigma = np.tile(tiled_sigma_pattern, (self.n_items, 1))
            elif self.sigma.ndim == 1 and self.n_items == 1:
                 tiled_sigma_pattern = np.tile(self.sigma, num_repeats)[:n_time_steps]
                 tiled_sigma = tiled_sigma_pattern.reshape(1, n_time_steps)
            else:
                tiled_sigma = np.tile(self.sigma, (1, num_repeats))[:, :n_time_steps]

            if self.seasonal_factor is not None:
                tiled_seasonal = np.tile(self.seasonal_factor, num_repeats)[:n_time_steps]
                season_t = tiled_seasonal[np.newaxis, :] # Add axis for broadcasting (n_items, n_time_steps)
                effective_means = tiled_mu * season_t
            else:
                effective_means = tiled_mu # No seasonal adjustment

            demand_floats = self.rng.normal(loc=effective_means, scale=tiled_sigma)

            if self.name_distribution == 'normal_truncated_at_zero':
                demand_non_negative = np.maximum(0, demand_floats)
            else: # 'normal' allows negative if not explicitly truncated (though often implies non-negative contextually)
                demand_non_negative = demand_floats # Keep as is, rounding might still make it 0

            demand_integers = np.round(demand_non_negative).astype(int)
            # For 'normal_truncated_at_zero', ensure values are truly >= 0 after rounding if any edge case float made it <0
            if self.name_distribution == 'normal_truncated_at_zero':
                demand_integers[demand_integers < 0] = 0

            return demand_integers

        elif self.name_distribution == 'poisson': # ADDED POISSON
            num_repeats = math.ceil(n_time_steps / self.pattern_length)

            # Tile lambda_param
            if self.lambda_param.ndim == 1 and self.n_items > 1:
                tiled_lambda_pattern = np.tile(self.lambda_param, num_repeats)[:n_time_steps]
                tiled_lambda = np.tile(tiled_lambda_pattern, (self.n_items, 1))
            elif self.lambda_param.ndim == 1 and self.n_items == 1:
                 tiled_lambda_pattern = np.tile(self.lambda_param, num_repeats)[:n_time_steps]
                 tiled_lambda = tiled_lambda_pattern.reshape(1, n_time_steps)
            else: # lambda_param is (n_items, pattern_length)
                tiled_lambda = np.tile(self.lambda_param, (1, num_repeats))[:, :n_time_steps]

            if self.seasonal_factor is not None:
                tiled_seasonal = np.tile(self.seasonal_factor, num_repeats)[:n_time_steps]
                season_t = tiled_seasonal[np.newaxis, :]
                effective_lambdas = tiled_lambda * season_t
            else:
                effective_lambdas = tiled_lambda
            
            # Ensure lambdas are non-negative for Poisson
            effective_lambdas_non_negative = np.maximum(0, effective_lambdas)
            
            return self.rng.poisson(lam=effective_lambdas_non_negative, size=output_shape)


        elif self.name_distribution == 'discrete_uniform':
            return self.rng.integers(
                low=self.low,
                high=self.high, # exclusive
                size=output_shape
            )

        elif self.name_distribution == 'binomial':
            return self.rng.binomial(
                n=self.n_binom,
                p=self.p_binom,
                size=output_shape
            )

        elif self.name_distribution == 'probability_mass_function':
             return self.rng.choice(
                a=self.pmf_vals,
                p=self.pmf_probs,
                size=output_shape,
                replace=True
            )
        else:
             raise ValueError(f"Internal error: Unsupported distribution {self.name_distribution}")