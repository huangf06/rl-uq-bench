"""
Gaussian observation noise wrapper for Gymnasium environments.
"""

from typing import Union, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType


class GaussianObsNoise(gym.ObservationWrapper):
    """
    Add i.i.d. Gaussian noise to observations for robustness testing.
    
    High-performance implementation that avoids unnecessary memory copies.
    
    Parameters
    ----------
    env : gym.Env
        Environment to wrap
    noise_std : float or array-like
        Noise standard deviation. Can be scalar (same noise for all dimensions) 
        or array (independent noise per dimension)
    clip : bool, optional
        Whether to clip noisy observations to original space bounds, default True
    seed : int, optional
        Random number generator seed, default None
    inplace : bool, optional
        Whether to modify observations in-place for maximum performance, default False
        WARNING: Only use if you're certain upstream code won't reuse observation arrays
    rng : np.random.Generator, optional
        Custom random number generator, default None (creates new one)
        
    Notes
    -----
    - Does not change observation_space, only modifies returned values
    - For use with VecNormalize: recommended order is VecNormalize first, then noise
      (this preserves noise characteristics after normalization)
    - Only supports Box observation spaces
    - Uses np.random.Generator for better performance and thread safety
    
    Examples
    --------
    >>> import gymnasium as gym
    >>> from wrappers import GaussianObsNoise
    >>> 
    >>> env = gym.make("Pendulum-v1")
    >>> env = GaussianObsNoise(env, noise_std=0.1, clip=True)
    >>> 
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """
    
    def _derive_offset(self) -> int:
        """Generate unique offset for this wrapper instance to avoid seed collisions in VecEnv."""
        return hash(self) & 0xFFFF  # 16-bit mask for reasonable offset range
        
    def __init__(
        self,
        env: gym.Env,
        noise_std: Union[float, np.ndarray],
        clip: bool = True,
        seed: Optional[int] = None,
        inplace: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(env)
        
        # Validate observation space type
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"GaussianObsNoise only supports Box observation spaces, "
                f"got {type(env.observation_space)}"
            )
        
        self.obs_space = env.observation_space
        self.clip = clip
        self.inplace = inplace
        
        # Process noise_std parameter with consistent dtype
        if np.isscalar(noise_std):
            if noise_std < 0:
                raise ValueError(f"noise_std must be non-negative, got {noise_std}")
            self.noise_std = float(noise_std)
            self._is_scalar_noise = True
        else:
            noise_std = np.asarray(noise_std, dtype=np.float32)
            if np.any(noise_std < 0):
                raise ValueError(f"noise_std must be non-negative, got {noise_std}")
            if noise_std.shape != self.obs_space.shape:
                raise ValueError(
                    f"noise_std shape {noise_std.shape} does not match "
                    f"observation space shape {self.obs_space.shape}. "
                    f"Expected shape: {self.obs_space.shape}"
                )
            self.noise_std = noise_std
            self._is_scalar_noise = False
        
        # Precompute clipping bounds if needed
        if self.clip:
            self.obs_low = self.obs_space.low
            self.obs_high = self.obs_space.high
            # Check for finite bounds
            self._has_finite_bounds = (
                np.isfinite(self.obs_low).all() and 
                np.isfinite(self.obs_high).all()
            )
        else:
            self._has_finite_bounds = False
        
        # Setup random number generator with unique offset for VecEnv compatibility
        if rng is not None:
            self._rng = rng
        else:
            # Use instance hash to ensure different noise sequences in VecEnv
            offset = self._derive_offset()
            effective_seed = (seed + offset) if seed is not None else None
            self._rng = np.random.default_rng(effective_seed)
            
    def _sample_noise(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Sample noise with specified shape and dtype."""
        if self._is_scalar_noise:
            # Generate noise and cast to required dtype
            noise = self._rng.normal(0.0, self.noise_std, size=shape)
            if noise.dtype != dtype:
                noise = noise.astype(dtype, copy=False)
            return noise
        else:
            # For array noise_std, sample and cast if needed
            noise = self._rng.normal(0.0, self.noise_std)
            if noise.dtype != dtype:
                noise = noise.astype(dtype, copy=False)
            return noise
        
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observation."""
        # Generate noise with matching dtype
        noise = self._sample_noise(observation.shape, observation.dtype)
        
        # Apply noise (in-place or create new array)
        if self.inplace:
            # Modify observation in-place for maximum performance
            target = observation
            np.add(observation, noise, out=target)
        else:
            # Create new array (safer default)
            target = observation + noise
        
        # Clip if enabled and bounds are finite (use in-place operation)
        if self.clip and self._has_finite_bounds:
            np.clip(target, self.obs_low, self.obs_high, out=target)
        
        return target
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment and apply noise to initial observation."""
        if seed is not None:
            # Use instance-specific offset to avoid seed collisions in VecEnv
            offset = self._derive_offset()
            effective_seed = seed + offset
            self._rng = np.random.default_rng(effective_seed)
        return super().reset(seed=seed, options=options)
    
    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}"
            f"(noise_std={self.noise_std}, clip={self.clip}, inplace={self.inplace}, "
            f"dtype={self.obs_space.dtype.name})>"
        )