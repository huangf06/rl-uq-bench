# Custom Gymnasium Wrappers

Custom environment wrappers for reinforcement learning.

## GaussianObsNoise

High-performance Gaussian observation noise wrapper for testing agent robustness to noisy observations.

### Features

- **High-performance implementation**: Avoids unnecessary memory copies and dtype conversions
- **Flexible noise configuration**: Supports scalar or array noise standard deviations
- **Optional clipping**: Choice to clip noisy observations to original bounds
- **Preserves observation space**: Only modifies returned values, not `observation_space`
- **Random seed support**: Supports reproducible noise generation
- **VecEnv compatible**: Works seamlessly with vectorized environments

### Usage

```python
import gymnasium as gym
from wrappers import GaussianObsNoise

# Basic usage
env = gym.make("Pendulum-v1")
env = GaussianObsNoise(env, noise_std=0.1, clip=True)

# Different noise per dimension
noise_std_array = np.array([0.05, 0.05, 0.2])
env = GaussianObsNoise(env, noise_std=noise_std_array, clip=True)
```

### Integration with VecNormalize

**Recommended wrapping order:**

1. Base environment
2. **VecNormalize** (normalize first)
3. **GaussianObsNoise** (add noise after normalization)

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    env = gym.make("Pendulum-v1")
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env)  # Normalize first
# Then add noise wrapper to individual environments if needed
```

**Alternative: Per-environment noise before vectorization:**

```python
def make_env():
    env = gym.make("Pendulum-v1")
    env = GaussianObsNoise(env, noise_std=0.1)  # Add noise first
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env)  # Then normalize
```

Both approaches are valid, but normalizing first preserves noise characteristics better.

### Parameters

- `env`: Gymnasium environment (Box observation space only)
- `noise_std`: Noise standard deviation (scalar or array matching observation shape)
- `clip`: Whether to clip noisy observations to original bounds, default `True`
- `seed`: Random number generator seed, default `None`

### Performance

- **Overhead**: ~9% for typical environments (1000 steps benchmark)
- **Memory**: Optimized to minimize allocations and dtype conversions
- **Dtype preservation**: Maintains original observation dtype

### Notes

- Only supports `gym.spaces.Box` observation spaces
- When `clip=True` and observation space has infinite bounds, no clipping is performed
- Test different noise levels before training to assess impact
- Zero noise (`noise_std=0`) is supported and will pass through observations unchanged 