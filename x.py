import cupy as cp
import numpy as np

from aspire.abinitio.commonline_sync3n import _signs_times_v_cupy, _signs_times_v_host
from aspire.utils import all_pairs

n = 4
n_pairs = n * (n - 1) // 2
_, _pairs_to_linear = all_pairs(n, return_map=True)

vec = np.ones(n_pairs, dtype=np.float64)
# Rijs = np.random.randn(n_pairs*3*3).astype(dtype=np.float64)
Rijs = np.arange(n_pairs * 3 * 3).reshape(n_pairs, 3, 3).astype(dtype=np.float64)


new_vec = _signs_times_v_cupy(n, Rijs, vec, J_weighting=None, _ALTS=None)
print("gpu\n", new_vec)


new_vec_host = _signs_times_v_host(n, Rijs, vec, J_weighting=None, _ALTS=None, _pairs_to_linear=_pairs_to_linear)
print("host\n", new_vec_host)
