"""
=================
Micrograph Source
=================

This tutorial will demonstrate how to set up and use ASPIRE's ``MicrographSimulation`` class.
"""

import numpy as np

from aspire.image import Image
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import MicrographSimulation, Simulation

# %%
# Creating a Micrograph Source
# ----------------------------
# A ``MicrographSimulation`` is populated with particle projections via a ``Simulation``, so we'll begin by creating a ``Simulation`` and passing it into our ``MicrographSimulation``

# Let's create our Simulation with a particle box size of 64 and one volume.
n_particles_per_micrograph = 4
n_micrographs = 4

sim = Simulation(
    L=64,
    n=n_particles_per_micrograph * n_micrographs,
    seed=1234,
    C=1,
    amplitudes=1,
    offsets=0,
)

# %%
# We'll pass our ``Simulation`` as an argument and give our ``MicrographSimulation`` other arguments.
# In this example, our MicrographSimulation has 4 micrographs of size 500, each with 4 particles.
src = MicrographSimulation(
    sim,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
    seed=1234,
)

# Plot the Micrographs
src.images[:].show()

# %%
# CTF Filters
# -----------
# By default, no CTF corruption is configured. To apply CTF filters, we have to pass them as arguments to the Simulation.

# Create our CTF Filter and add it to a list
ctfs = [
    RadialCTFFilter(pixel_size=4, voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0),
]

# Pass the CTFs into the Simulation, and create a MicrographSimulation using the same arguments as before
sim = Simulation(
    L=64,
    n=n_particles_per_micrograph * n_micrographs,
    C=1,
    amplitudes=1,
    offsets=0,
    unique_filters=ctfs,
    seed=1234,
)
src = MicrographSimulation(
    sim,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
    seed=1234,
)

# Plot the micrographs
src.images[:].show()

# %%
# Noise
# -----
# By default, no noise corruption is configured. To apply noise, we have to pass them as arguments to the ``MicrographSimulation``

# Create our noise using WhiteNoiseAdder
noise = WhiteNoiseAdder(1e-3, seed=1234)

# Let's add noise to our MicrographSimulation using the noise_adder argument
src = MicrographSimulation(
    sim,
    noise_adder=noise,
    particles_per_micrograph=4,
    micrograph_size=500,
    micrograph_count=4,
    seed=1234,
)

# Plot the micrographs
src.images[:].show()

# %%
# We can also plot the un-noisy micrographs using the ``clean_micrographs`` accessor
src.clean_images[:].show()

# %%
# Interparticle Distance
# ----------------------
# By default, particle distance is set to avoid collisions.
# We can use the ``interparticle_distance`` argument to control the minimum distance between particle centers
# However, setting this argument too large may generate insufficient centers.

# Let's increase the number of particles to show overlap.
# Create a new simulation to meet the minimum required amount of projections.
n_particles_per_micrograph = 20

sim = Simulation(
    L=64,
    n=n_particles_per_micrograph * n_micrographs,
    C=1,
    amplitudes=1,
    offsets=0,
    unique_filters=ctfs,
)


# Set the interparticle distance to 1, which adds at least one pixel of separation between center and allows particles to collide.
src = MicrographSimulation(
    sim,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
)

# Plot the micrographs
src.images[:].show()

# %%
# Boundary
# --------
# By default, the boundary is set to half of the particle width, which will completely contain every particle inside the micrograph.
# Setting ``boundary=0`` will allow particles to be placed along the edges.
# Positive values (measured in pixels) move the boundaries inward, while negative values move the boundaries outward.

# Create a micrograph with a negative boundary, allowing particles to generate outward.
out_src = MicrographSimulation(
    sim,
    boundary=-20,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
)

# Plot the micrographs
out_src.images[:].show()

# %%
# Particle Indices
# ----------------
# Each particle comes from the simulation, and has its own index relative to its ``Simulation`` and micrograph.
# Let's choose four random numbers as our global (Simulation) particle indices from ``test_micrograph=1``.
test_micrograph = 1
n_particles = 4
local_particle_indices = np.random.choice(n_particles_per_micrograph, n_particles)
print(f"Local particle indices: {local_particle_indices}")

# %%
# We can obtain the images from our MicrographSimulation by retrieving their centers and plotting the boundary boxes.
centers = np.zeros((n_particles, 2), dtype=int)
for i in range(n_particles):
    centers[i] = src.centers[test_micrograph][local_particle_indices[i]]

# Let's use the particles' centers and sizes to perform "perfect particle picking" on this test micrograph.
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
Image(micrograph_picked_particles)[:].show()

# %%
# Let's find the images from the ``Simulation`` using the ``get_particle_indices`` method to retrieve their global indices.
global_particle_indices = np.zeros((n_particles), dtype=int)
for i in range(n_particles):
    global_particle_indices[i] = src.get_particle_indices(
        test_micrograph, local_particle_indices[i]
    )

# Plot the simulation's images
sim.images[global_particle_indices].show()

# %%
# We can check if these global indices match our local particle indices with the ``get_micrograph_index`` method.
check_local_indices = np.zeros((n_particles), dtype=int)
for i in range(n_particles):
    # Get each particle's corresponding micrograph index and local particle index
    micrograph_index, check_local_indices[i] = src.get_micrograph_index(
        global_particle_indices[i]
    )
    assert micrograph_index == 1
np.testing.assert_array_equal(local_particle_indices, check_local_indices)
print(f"Local particle indices: {check_local_indices}")
