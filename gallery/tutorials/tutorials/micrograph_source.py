"""
=====================
Micrograph Sources
=====================

This tutorial will demonstrate how to set up and use ASPIRE's
``MicrographSource`` classes.
"""

import os
import tempfile

import numpy as np

from aspire.source import ArrayMicrographSource

# %%
# Overview
# --------
# ``MicrographSource`` is an abstract class which provides access to
# three distinct subclasses.  The first two are
# ``ArrayMicrographSource`` and ``DiskMicrographSource`` which provide
# access to array and disk backed micrograph data respectively.
# ``MicrographSimulation`` takes a volume and generates projection
# images which are aggregated into synthetic microgaphs.  The following
# illustrates an overview of the interfaces, and the tutorial will go
# on to demonstrate common operations for each class.

# %%
#
#  .. mermaid::
#
#    classDiagram
#        class MicrographSource{
#            micrograph_count: int
#            micrograph_size: int
#            dtype: np.dtype
#            +asnumpy()
#            +dtype
#            +len()
#            +repr()
#            +images[]
#            +micrograph_count
#            +micrograph_size
#            +save()
#            +show()
#         }
#
#        class ArrayMicrographSource{
#            micrographs: np.ndarray
#         }
#
#        class DiskMicrographSource{
#            micrographs_path: str, Path, or list
#         }
#
#        class MicrographSimulation{
#            volume: Volume
#            micrograph_size: Optional, int
#            micrograph_count: Optional, int
#            particles_per_micrograph: Optional, int
#            particle_amplitudes: Optional, np.ndarray
#            projection_angles: Optional, np.ndarray
#            seed: Optional, int
#            ctf_filters: Optional, list
#            noise_adder: Optional, NoiseAdder
#            boundary: Optional, int
#            interparticle_distance: Optional, int
#            +boundary
#            +centers
#            +ctf_filters
#            +clean_images[]
#            +filter_indices
#            +get_micrograph_index()
#            +get_particle_index()
#            +interparticle_distance
#            +noise_adder
#            +simulation
#            +particle_amplitudes
#            +particle_box_size
#            +particle_per_micrograph
#            +projection_angles
#            +total_particle_count
#            +volume
#         }
#
#         MicrographSource <|-- ArrayMicrographSource
#         MicrographSource <|-- DiskMicrographSource
#         MicrographSource <|-- MicrographSimulation
#         MicrographSimulation o-- Volume
#         MicrographSimulation *-- CTFFilter
#         MicrographSimulation *-- NoiseAdder

# %%
# Creating an ArrayMicrographSource
# ---------------------------------
# An ``ArrayMicrographSource`` is populated with an array.  For this
# demonstration, random data will initialize the object,
# then this data will be saved off for use in the next example
# (which loads data from files).


# Create an (2,512,512) array of data.
# This represents two (512,512) micrographs.
mgs_np = np.random.rand(2, 512, 512)

# Construct the source
src = ArrayMicrographSource(mgs_np)

# Create a tmp dir for saving the data to.
# This just for ensuring the tutorial script is portable,
tmp_dir = tempfile.TemporaryDirectory()

# Save the data as multiple MRC files
# This method returns a file_list,
# which might be useful for loading or other operations.
file_list = src.save(tmp_dir.name)

# %%
# Creating a DiskMicrographSource
# -------------------------------
# A ``DiskMicrographSource`` is populated with str or list
# representing the location of MRC files.

from aspire.source import DiskMicrographSource

# Load files in directory
src = DiskMicrographSource(tmp_dir.name)

# Load files from a list
src = DiskMicrographSource(file_list)

# %%
# Creating a Micrograph Simulation
# --------------------------------
# A ``MicrographSimulation`` is populated with particle projections
# from a ``Volume``, so we'll begin by generating a ``Volume``.

from aspire.source import MicrographSimulation
from aspire.volume import AsymmetricVolume

# Generate one (100,100,100) ``Volume``.
vol = AsymmetricVolume(
    L=100,
    C=1,
    pixel_size=4,
    seed=1234,
    dtype=np.float32,
).generate()

# %%
# We'll pass our ``Volume`` as an argument and configure our
# ``MicrographSimulation``.  In this example, the
# ``MicrographSimulation`` has 4 micrographs of size 1024, each with 10
# particles.

n_particles_per_micrograph = 10
n_micrographs = 3

src = MicrographSimulation(
    vol,
    particles_per_micrograph=n_particles_per_micrograph,
    particle_amplitudes=1,
    micrograph_size=1024,
    micrograph_count=n_micrographs,
    seed=1234,
)

# Plot the micrographs
src.images[:].show()

# %%
# CTF Filters
# -----------
# By default, no CTF corruption is configured. To apply CTF filters,
# we have to pass them as arguments to the ``MicrographSimulation``.
# It is possible to apply a single CTF, different CTF per-micrograph
# or different CTF per-particle by configuring a list of matching size.

from aspire.operators import RadialCTFFilter

# Create our CTF Filter and add it to a list.
# This configuration will apply the same CTF to all particles.
ctfs = [
    RadialCTFFilter(voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0),
]

src = MicrographSimulation(
    vol,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=1024,
    micrograph_count=n_micrographs,
    ctf_filters=ctfs,
    seed=1234,
)

# Plot the micrographs
src.images[:].show()

# %%
# Noise
# -----
# By default, no noise corruption is configured.
# To apply noise, pass a ``NoiseAdder`` to ``MicrographSimulation``.

from aspire.noise import WhiteNoiseAdder

# Create our noise using WhiteNoiseAdder
noise = WhiteNoiseAdder(4e-3, seed=1234)

# Add noise to our MicrographSimulation using the noise_adder argument
src = MicrographSimulation(
    vol,
    noise_adder=noise,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=1024,
    micrograph_count=n_micrographs,
    ctf_filters=ctfs,
    seed=1234,
)

# Plot the micrographs
src.images[:].show()

# %%
# Plot the clean micrographs using the ``clean_images`` accessor.
src.clean_images[:].show()

# %%
# Interparticle Distance
# ----------------------
# By default, particle distance is set to avoid collisions.
# We can use the ``interparticle_distance`` argument to control the
# minimum distance between particle centers.
# However, setting this argument too large may generate insufficient centers.

# Let's increase the number of particles to show overlap.
n_particles_per_micrograph = 50

# Set the interparticle distance to 1, which adds at least one pixel
# of separation between center and allows particles to collide.
src = MicrographSimulation(
    vol,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=1024,
    micrograph_count=n_micrographs,
    ctf_filters=ctfs,
)

# Plot the micrographs
src.images[:].show()

# %%
# Boundary
# --------
# By default, the boundary is set to half of the particle width,
# which will completely contain every particle inside the micrograph.
# Setting ``boundary=0`` will allow particles to be placed along the edges.
# Positive values (measured in pixels) move the boundaries inward,
# while negative values move the boundaries outward.

# Create a micrograph with a negative boundary, allowing particles to
# generate outward.
out_src = MicrographSimulation(
    vol,
    boundary=-20,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=1024,
    micrograph_count=n_micrographs,
    ctf_filters=ctfs,
)

# Plot the micrographs
out_src.images[:].show()

# %%
# Particle Indices
# ----------------
# Each particle comes from a ``Simulation`` internal to
# ``MicrographSimulation``.  This simulation can be accessed directly
# by the attribute ``MicrographSimulation.simulation``.  A map is
# provided between each particle's indexing relative to that
# ``Simulation`` and micrograph based indexing.  This relationship is
# demonstrated below.

# Let's choose four random numbers as our global (``Simulation``)
# particle indices from ``test_micrograph=1``.
test_micrograph = 1
n_particles = 3
local_particle_indices = np.random.choice(n_particles_per_micrograph, n_particles)
print(f"Local particle indices: {local_particle_indices}")

# %%
# We can obtain the individual particle images from our
# ``MicrographSimulation`` by retrieving their centers and plotting
# the boundary boxes.
centers = np.zeros((n_particles, 2), dtype=int)
for i in range(n_particles):
    centers[i] = src.centers[test_micrograph][local_particle_indices[i]]

# Let's use the particles' centers and sizes to perform "perfect
# particle picking" on this test micrograph.
p_size = src.particle_box_size
micrograph_picked_particles = np.zeros(
    (
        n_particles,
        src.particle_box_size,
        src.particle_box_size,
    )
)

for i, center in enumerate(centers):
    x, y = center[0], center[1]
    # Calculate the square of the particle
    particle = src.clean_images[test_micrograph].asnumpy()[0][
        x - p_size // 2 : x + p_size // 2, y - p_size // 2 : y + p_size // 2
    ]
    micrograph_picked_particles[i] = particle

# Let's plot and look at the particles!
from aspire.image import Image

Image(micrograph_picked_particles)[:].show()

# %%
# .. note::
#     There may be overlap with nearby particles in the above images.
#     To reduce overlap, increase ``interparticle_distance``.


# %%
# Let's find the images from the ``Simulation`` using the
# ``get_particle_indices`` method to retrieve their global indices.
global_particle_indices = np.zeros((n_particles), dtype=int)
for i in range(n_particles):
    global_particle_indices[i] = src.get_particle_indices(
        test_micrograph, local_particle_indices[i]
    )

# Plot the simulation's images
src.simulation.images[global_particle_indices].show()

# %%
# We can check if these global indices match our local particle
# indices with the ``get_micrograph_index`` method.
check_local_indices = np.zeros((n_particles), dtype=int)
for i in range(n_particles):
    # Get each particle's corresponding micrograph index and local particle index
    micrograph_index, check_local_indices[i] = src.get_micrograph_index(
        global_particle_indices[i]
    )
    assert micrograph_index == 1
np.testing.assert_array_equal(local_particle_indices, check_local_indices)
print(f"Local particle indices: {check_local_indices}")

# %%
# Saving a MicrographSimulation
# -----------------------------
# In addition to saving the raw MRC files, ``MicrographSimulation``
# populates STAR files with the particle centers, particle box size
# (``rlnImageSize``), and projection rotations.  Additionally, CTF
# parameters are saved when CTF is used in the simulation.  Each
# micrograph will have a corresponidng STAR file.  The collection of
# these files are returned from ``MicrographSimulation.save`` as a
# list of tuples which is designed to work directly with
# ``CentersCoordinateSource``.

from aspire.source import CentersCoordinateSource

# Save the simulation
results = src.save(os.path.join(tmp_dir.name, "mg_sim"))

# %%

# Review the resulting files
print(results)

# %%

# Review the example STAR file contents
with open(results[0][1], "r") as f:
    print(f.read())

# %%

img_src = CentersCoordinateSource(results, src.particle_box_size)
# Show the first five images from the image source.
img_src.images[:3].show()

# Cleanup the tmp_dir
tmp_dir.cleanup()
