"""
Simulated Stack → RELION Reconstruction
=======================================

This experiment shows how to:

1. build a synthetic dataset with ASPIRE,
2. write the stack via ``ImageSource.save`` so RELION can consume it, and
3. call :code:`relion_reconstruct` on the saved STAR file.
"""

# %%
# Imports
# -------

import logging
from pathlib import Path

import numpy as np

from aspire.downloader import emdb_2660
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import RelionSource, Simulation

logger = logging.getLogger(__name__)


# %%
# Configuration
# -------------
# We set a few parameters to initialize the Simulation.
# You can safely alter ``n_particles`` (or change the voltages, etc.) when
# trying this interactively; the defaults here are chosen for demonstrative purposes.

output_dir = Path("relion_save_demo")
output_dir.mkdir(exist_ok=True)

n_particles = 512
snr = 0.25
voltages = np.linspace(200, 300, 3)  # kV settings for the radial CTF filters
star_path = output_dir / f"sim_n{n_particles}.star"


# %%
# Volume and Filters
# ------------------
# Start from the EMDB-2660 ribosome map and build a small set of radial CTF filters
# that RELION will recover as optics groups.

vol = emdb_2660()
ctf_filters = [RadialCTFFilter(voltage=kv) for kv in voltages]


# %%
# Simulate, Add Noise, Save
# -------------------------
# Initialize the Simulation:
# mix the CTFs across the stack, add white noise at a target SNR,
# and write the particles and metadata to a RELION-compatible STAR/MRC stack.

sim = Simulation(
    n=n_particles,
    vols=vol,
    unique_filters=ctf_filters,
    noise_adder=WhiteNoiseAdder.from_snr(snr),
)
sim.save(star_path, overwrite=True)


# %%
# Running ``relion_reconstruct``
# ------------------------------
# ``relion_reconstruct`` is an external RELION command, so we just show the call.
# Run this in a RELION-enabled shell after generating the STAR file above.

relion_cmd = [
    "relion_reconstruct",
    "--i",
    str(star_path),
    "--o",
    str(output_dir / "relion_recon.mrc"),
    "--ctf",
]

print(" ".join(relion_cmd))

