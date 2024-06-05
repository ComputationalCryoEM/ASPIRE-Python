import os
import pickle
from pprint import pprint
from time import perf_counter, time

import matplotlib.pyplot as plt
import numpy as np

from aspire.basis import FFBBasis2D, FLEBasis2D
from aspire.downloader import emdb_2660
from aspire.noise import WhiteNoiseAdder
from aspire.source import ArrayImageSource, Simulation

# Download and cache volume map
vol = emdb_2660().astype(np.float64)  # doubles
cached_image_fn = "simulated_images.npy"

if os.path.exists(cached_image_fn):
    print(f"Loading cached image source from {cached_image_fn}.")
    sim = ArrayImageSource(np.load(cached_image_fn))
else:
    print("Generating Simulated Datatset")
    sim = Simulation(
        n=512, C=1, vols=vol, noise_adder=WhiteNoiseAdder.from_snr(0.1)
    ).cache()
    print(f"Saving to {cached_image_fn}")
    np.save(cached_image_fn, sim.images[:].asnumpy())


TIMES = {}
for L in [32, 64, 128, 256]:
    print(f"Begin L={L}")
    src = sim.downsample(L)
    imgs = src.images[:]
    TIMES[L] = {}
    for basis_type in [FFBBasis2D, FLEBasis2D]:
        # Construct basis
        TIMES[L][basis_type.__name__] = {}
        basis = basis_type(L, dtype=src.dtype)

        # Time expanding into basis
        tic = perf_counter()
        coef = basis.evaluate_t(imgs)
        toc = perf_counter()
        TIMES[L][basis_type.__name__]["evaluate_t"] = toc - tic

        # Time expanding back into images
        tic = perf_counter()
        _ = coef.evaluate()
        toc = perf_counter()
        TIMES[L][basis_type.__name__]["evaluate"] = toc - tic


pprint(TIMES)


with open(f"benchmark_{int(time())}.pkl", "wb") as fh:
    pickle.dump(TIMES, fh)
