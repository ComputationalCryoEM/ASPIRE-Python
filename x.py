from aspire.abinitio.commonline_sync3n import _signs_times_v_cupy
import numpy as np
import cupy as cp

n = 7
n_pairs = n*(n-1)//2
vec = np.ones(n_pairs, dtype=np.float64)
new_vec = np.zeros(n_pairs, dtype=np.float64)
#Rijs = np.random.randn(n_pairs*3*3).astype(dtype=np.float64)
Rijs = np.arange(n_pairs*3*3).astype(dtype=np.float64)


new_vec = _signs_times_v_cupy(n, Rijs, vec, J_weighting=None, _ALTS=None, _signs_confs=None)

print(new_vec)
