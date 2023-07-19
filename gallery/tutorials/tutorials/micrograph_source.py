"""
=================
Micrograph Source
=================

This tutorial will demonstrate how to set up and use ASPIRE's ``MicrographSource`` class.
"""

import numpy as np

from aspire.image import Image
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import MicrographSource, Simulation

# %%
# Creating a Micrograph Source
# ----------------------------
# A ``MicrographSource`` is populated with particle projections via a ``Simulation``, so we'll begin by creating a ``Simulation`` and passing it into our ``MicrographSource``

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
# We'll pass our ``Simulation`` as an argument and give our ``MicrographSource`` other arguments.
# In this example, our MicrographSource has 4 micrographs of size 500, each with 4 particles.
src = MicrographSource(
    sim,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
    seed=1234,
)

# Plot the Micrographs
src.micrographs[:].show()

# %%
# CTF Filters
# -----------
# By default, no CTF corruption is configured. To apply CTF filters, we have to pass them as arguments to the Simulation.

# Create our CTF Filter and add it to a list
ctfs = [
    RadialCTFFilter(pixel_size=4, voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0),
]

# Pass the CTFs into the Simulation, and create a MicrographSource using the same arguments as before
sim = Simulation(
    L=64,
    n=n_particles_per_micrograph * n_micrographs,
    C=1,
    amplitudes=1,
    offsets=0,
    unique_filters=ctfs,
    seed=1234,
)
src = MicrographSource(
    sim,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
    seed=1234,
)

# Plot the micrographs
src.micrographs[:].show()

# %%
# Noise
# -----
# By default, no noise corruption is configured. To apply noise, we have to pass them as arguments to the ``MicrographSource``

# Create our noise using WhiteNoiseAdder
noise = WhiteNoiseAdder(1e-3, seed=1234)

# Let's add noise to our MicrographSource using the noise_adder argument
src = MicrographSource(
    sim,
    noise_adder=noise,
    particles_per_micrograph=4,
    micrograph_size=500,
    micrograph_count=4,
    seed=1234,
)

# Plot the micrographs
src.micrographs[:].show()

# %%
# We can also plot the un-noisy micrographs using the ``clean_micrographs`` accessor
src.clean_micrographs[:].show()

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
src = MicrographSource(
    sim,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
)

# Plot the micrographs
src.micrographs[:].show()

# %%
# Boundary
# --------
# By default, the boundary is set to half of the particle width , which will completely contain every particle inside the micrograph.
# Setting ``boundary=0`` will allow particle centers to generate along the edges.
# Positive values (measured in pixels) move the boundaries inward, while negative values move the boundaries outward.

# Create a micrograph with a negative boundary, allowing particles to generate outward.
out_src = MicrographSource(
    sim,
    boundary=-20,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=n_particles_per_micrograph,
    micrograph_size=500,
    micrograph_count=n_micrographs,
)

# Plot the micrographs
out_src.micrographs[:].show()

# %%
# Particle IDs
# ------------
# Each particle comes from the simulation, and has its own ID relative to its ``Simulation`` and micrograph.
# Let's choose 4 random numbers, as our global (Simulation) particle IDs from ``test_micrograph=1``.
test_micrograph = 1
n_particles = 4
local_particle_ids = np.random.choice(n_particles_per_micrograph, n_particles)
print(f"Local particle IDS: {local_particle_ids}")

# %%
# We can obtain the images from our MicrographSource by retrieving their centers and plotting the boundary boxes.
centers = np.zeros((n_particles, 2), dtype=int)
for i in range(n_particles):
    centers[i] = src.centers[test_micrograph][local_particle_ids[i]]

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
    particle = src.clean_micrographs[test_micrograph].asnumpy()[0][
        x - p_size // 2 : x + p_size // 2, y - p_size // 2 : y + p_size // 2
    ]
    micrograph_picked_particles[i] = particle

# Let's plot and look at the particles!
Image(micrograph_picked_particles)[:].show()

# %%
# The simulated particles are inverted from the MRC due to convention, as we're trying to measure interference.
# Let's find the images from the ``Simulation`` using the ``get_particle`` method to retrieve their global IDs.
global_particle_ids = np.zeros((n_particles), dtype=int)
for i in range(n_particles):
    global_particle_ids[i] = src.get_particle(test_micrograph, local_particle_ids[i])

# Plot the simulation's images
sim.images[global_particle_ids].show()

# %%
# We can check if these global IDs match our local particle IDs with the ``get_micrograph`` method.
check_local_ids = np.zeros((n_particles), dtype=int)
for i in range(n_particles):
    # Get each particle's corresponding micrograph ID and local particle ID
    micrograph_id, check_local_ids[i] = src.get_micrograph(global_particle_ids[i])
    assert micrograph_id == 1
np.testing.assert_array_equal(local_particle_ids, check_local_ids)
print(f"Local particle IDs: {check_local_ids}")
