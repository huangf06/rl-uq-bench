import os
import numpy as np
import re

base = 'logs/multi_env_experiments/LunarLander-v3/uncertainty_degradation_noise0.000/qrdqn'
results = []
seeds = sorted([d for d in os.listdir(base) if d.startswith('seed_')])
print('QR-DQN σ=0.000 各种子训练用时（秒/小时）:')
for s in seeds:
    f = os.path.join(base, s, '0.monitor.csv')
    if os.path.exists(f):
        with open(f, 'r') as fin:
            lines = fin.readlines()
            # t_start
            t_start = None
            m = re.match(r'#\{"t_start": ([0-9\.]+),', lines[0])
            if m:
                t_start = float(m.group(1))
            # 最后一行的t
            for line in reversed(lines):
                if not line.startswith('#') and ',' in line:
                    parts = line.strip().split(',')
                    t = float(parts[-1])
                    break
            else:
                t = None
            if t_start is not None and t is not None:
                duration = t
                print(f'  {s}: {duration:.1f} 秒 / {duration/3600:.2f} 小时')
                results.append(duration)
results = np.array(results)
if len(results) > 0:
    print(f'均值: {np.nanmean(results):.1f} 秒 / {np.nanmean(results)/3600:.2f} 小时')
    print(f'最大: {np.nanmax(results):.1f} 秒 / {np.nanmax(results)/3600:.2f} 小时')
    print(f'最小: {np.nanmin(results):.1f} 秒 / {np.nanmin(results)/3600:.2f} 小时') 