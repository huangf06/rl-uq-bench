import os
import numpy as np

base = 'logs/multi_env_experiments/LunarLander-v3/uncertainty_degradation_noise0.000/qrdqn'
results = []
seeds = sorted([d for d in os.listdir(base) if d.startswith('seed_')])
print('QR-DQN σ=0.000 各种子最终分数:')
for s in seeds:
    f = os.path.join(base, s, 'evaluations.npz')
    if os.path.exists(f):
        arr = np.load(f)
        r = arr['results'].squeeze()
        # 取最后一个分数，若为数组则取均值
        if isinstance(r, np.ndarray):
            if r.ndim == 0:
                final = float(r)
            elif r.ndim == 1:
                final = float(r[-1])
            else:
                final = float(np.mean(r[-1]))
        else:
            final = float(r)
        print(f'  {s}: {final:.1f}')
        results.append(final)
results = np.array(results)
print(f'均值: {np.nanmean(results):.1f}')
print(f'标准差: {np.nanstd(results):.1f}')
print(f'最大: {np.nanmax(results):.1f}')
print(f'最小: {np.nanmin(results):.1f}') 