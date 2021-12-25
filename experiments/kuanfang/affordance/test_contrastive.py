import numpy as np


delta_t = 15
T = 75

t0 = 17
t1 = t0 + delta_t

t2_candidates = np.concatenate(
    [np.arange(0, t0 - delta_t), np.arange(t1 + delta_t, T)],
    axis=0)
print(t2_candidates)
print(len(t2_candidates))
