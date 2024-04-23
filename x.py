import pickle
import time
from collections import defaultdict

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from aspire.abinitio.commonline_sync3n import _signs_times_v_cupy, _signs_times_v_host
from aspire.utils import all_pairs


def time_test(n):
    n_pairs = n * (n - 1) // 2
    _, _pairs_to_linear = all_pairs(n, return_map=True)

    vec = np.ones(n_pairs, dtype=np.float64)
    # Rijs = np.random.randn(n_pairs*3*3).astype(dtype=np.float64)
    Rijs = np.arange(n_pairs * 3 * 3).reshape(n_pairs, 3, 3).astype(dtype=np.float64)

    tic0 = time.perf_counter()
    new_vec = _signs_times_v_cupy(n, Rijs, vec, J_weighting=False)
    tic1 = time.perf_counter()
    gpu_time = tic1 - tic0
    print("gpu\n", new_vec)

    tic2 = time.perf_counter()
    new_vec_host = _signs_times_v_host(
        n, Rijs, vec, J_weighting=False, _ALTS=None, _pairs_to_linear=_pairs_to_linear
    )
    tic3 = time.perf_counter()
    host_time = tic3 - tic2
    print("host\n", new_vec_host)

    print(f"\n\n\nSize:\t{n}")
    print("Allclose? ", np.allclose(new_vec_host, new_vec))
    print(f"gpu_time: {gpu_time}")
    print(f"host_time: {host_time}")
    speedup = host_time / gpu_time
    print(f"speedup: {speedup}")

    return host_time, gpu_time, speedup


def plotit(results):
    N = np.array(list(results.keys()))
    H = np.array([v["host"] for v in results.values()])
    G = np.array([v["gpu"] for v in results.values()])
    S = np.array([v["speedup"] for v in results.values()])

    plt.plot(N, H, label="host python")
    plt.plot(N, G, label="cuda")
    plt.title("Walltimes (s)")
    plt.legend()
    plt.show()
    plt.savefig("walltimes.png")
    plt.clf()

    plt.plot(N, S)
    plt.title("Speedup Ratio")
    plt.show()
    plt.savefig("speedups.png")
    plt.clf()


def main():
    results = defaultdict(dict)
    # too long...! for n in [4,16,64,100,128,200,256,512,1024,2048,3000, 4096, 10000]:
    # for n in [4,16]: # test
    for n in [4, 16, 64, 100, 128, 200, 512]:
        h, g, s = time_test(n)
        results[n]["host"] = h
        results[n]["gpu"] = g
        results[n]["speedup"] = s

        # save in case we cancel
        with open("saved_results.pkl", "wb") as f:
            pickle.dump(results, f)

    print()
    print(results)
    print()

    plotit(results)


time_test(64)
