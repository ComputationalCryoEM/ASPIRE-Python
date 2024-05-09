import os
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

host_fn = "benchmark_host.pkl"
gpu_fn = "benchmark_gpu0.pkl"


with open(host_fn, "rb") as fh:
    host_times = pickle.load(fh)

with open(gpu_fn, "rb") as fh:
    gpu_times = pickle.load(fh)

markers = {"FFBBasis2D": "8", "FLEBasis2D": "s"}

# Evaluate_t
Ls = list(host_times.keys())
for basis_type in markers.keys():
    plt.plot(
        Ls,
        [host_times[L][basis_type]["evaluate_t"] for L in Ls],
        marker=markers[basis_type],
        color="blue",
        label=basis_type + "-host",
    )
    plt.plot(
        Ls,
        [gpu_times[L][basis_type]["evaluate_t"] for L in Ls],
        marker=markers[basis_type],
        color="green",
        label=basis_type + "-gpu",
    )
plt.title("Basis `evaluate_t` Permformance - Batch of 512 Images")
plt.xlabel("Image Pixel L (LxL)")
plt.ylabel("Time (seconds)")
plt.legend()
plt.savefig("evaluate_t.png")
plt.show()

for basis_type in markers.keys():
    plt.plot(
        Ls,
        [host_times[L][basis_type]["evaluate"] for L in Ls],
        marker=markers[basis_type],
        color="blue",
        label=basis_type + "-host",
    )
    plt.plot(
        Ls,
        [gpu_times[L][basis_type]["evaluate"] for L in Ls],
        marker=markers[basis_type],
        color="green",
        label=basis_type + "-gpu",
    )
plt.title("Basis `evaluate` Permformance - Batch of 512 Images")
plt.xlabel("Image Pixel L (LxL)")
plt.ylabel("Time (seconds)")
plt.legend()
plt.savefig("evaluate.png")
plt.show()
