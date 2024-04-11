from aspire.abinitio.commonline_sync3n import _signs_times_v_cupy
from aspire.abinitio.commonline_sync3n import _signs_times_v_host
import numpy as np
import cupy as cp

n = 7
n_pairs = n*(n-1)//2
vec = np.ones(n_pairs, dtype=np.float64)
new_vec = np.zeros(n_pairs, dtype=np.float64)
#Rijs = np.random.randn(n_pairs*3*3).astype(dtype=np.float64)
Rijs = np.arange(n_pairs*3*3).reshape(n_pairs,3,3).astype(dtype=np.float64)


new_vec = _signs_times_v_cupy(n, Rijs, vec, J_weighting=None, _ALTS=None)

print("gpu\n")
print(new_vec)

new_vec_host = _signs_times_v_host(n, Rijs, vec, J_weighting=None, _ALTS=None)

print("host\n",new_vec_host)

