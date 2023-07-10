"""
=================
Micrograph Source
=================

This tutorial will demonstrate how to set up and use ASPIRE's ``MicrographSource`` class.
"""

import numpy as np
from aspire.image import Image
from aspire.source import Simulation
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import MicrographSource

# %%
# Creating a Micrograph Source
# ----------------------------
# We need to generate our projections by using a Simulation, which is a required argument. Then, we'll pass that Simulation into our MicrographSource and add the other arguments.
# You can also use your own Simulations as arguments for the MicrographSource.

# Let's create our Simulation with a particle box size of 30 and one volume.
sim = Simulation(L=64, n=4 * 4, seed=1234, C=1, amplitudes=1, offsets=0)

# %%
# We'll pass our Simulation as an argument and give our MicrographSource other arguments.
# In this example, our MicrographSource has 2 micrographs of size 500, each with 4 particles.
micrograph = MicrographSource(
    sim, particles_per_micrograph=4, micrograph_size=500, micrograph_count=4, seed=1234
)

# Plot the Micrographs
micrograph.micrographs[:5].show()

# %%
# CTF Filters
# -----------
# By default, no noise corruption is configured. To apply CTF filters, we have to pass them as arguments to the Simulation

# Create our list of CTF filters
ctfs = [
    RadialCTFFilter(pixel_size=4, voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0),
    RadialCTFFilter(pixel_size=4, voltage=200, defocus=10000, Cs=2.26, alpha=0.07, B=0),
    RadialCTFFilter(pixel_size=5, voltage=200, defocus=20000, Cs=2.26, alpha=0.07, B=0),
]

# Pass the CTFs into the Simulation, and create a MicrographSource using the same arguments as before
corrupted_sim = Simulation(
    L=64, n=4 * 4, seed=1234, C=1, amplitudes=1, offsets=0, unique_filters=ctfs
)
corrupted_micrograph = MicrographSource(
    corrupted_sim,
    particles_per_micrograph=4,
    micrograph_size=500,
    micrograph_count=4,
    seed=1234,
)

# Plot the Micrographs
corrupted_micrograph.micrographs[:5].show()

# %%
# Noise
# -----
# By default, no noise corruption is configured. To apply noise, we have to pass them as arguments to the Micrograph Source

# Create our noise using WhiteNoiseAdder
noise = WhiteNoiseAdder(1e-3, seed=1234)

# Pass the CTFs into the Simulation, and create a MicrographSource using the same arguments as before
noisy_micrograph = MicrographSource(
    corrupted_sim,
    noise_adder=noise,
    particles_per_micrograph=4,
    micrograph_size=500,
    micrograph_count=4,
    seed=1234,
)

# Plot the micrographs
noisy_micrograph.micrographs[:5].show()

# %%
# We can also plot the un-noisy micrographs using the ``clean_micrographs`` accessor
noisy_micrograph.clean_micrographs[:5].show()

# %%
# Interparticle Distance
# ----------------------
# By default, particle distance is set to avoid collisions.
# We can use this argument to control the minimum distance between particle centers
# However, setting this argument too large may generate insufficient centers.

# Let's increase the number of particles to show overlap.
# Create a new simulation to meet the minimum required amount of projections.
colliding_sim = Simulation(
    L=64, n=5 * 20, seed=1234, C=1, amplitudes=1, offsets=0, unique_filters=ctfs
)


# Set the interparticle distance to 1 so the particles can collide but not completely overlap.
colliding_micrograph = MicrographSource(
    colliding_sim,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=20,
    micrograph_size=500,
    micrograph_count=4,
    seed=1234,
)

# Plot the micrographs
colliding_micrograph.micrographs[:5].show()

# %%
# Boundary
# --------
# By default, the boundary is set to completely expose every particle in the micrograph.
# Setting ``boundary=0`` will allow particle centers to generate along the edges.
# Positive values move the boundaries inward, while negative values move the boundaries outward.

# Create a micrograph with a negative boundary, allowing particles to generate outward.
outbound_micrograph = MicrographSource(
    colliding_sim,
    boundary=-20,
    interparticle_distance=1,
    noise_adder=noise,
    particles_per_micrograph=20,
    micrograph_size=500,
    micrograph_count=4,
    seed=1234,
)

# Plot the micrographs
outbound_micrograph.micrographs[:5].show()

# %%
# Particle IDS
# ------------
# Each particle comes from the simulation, and has its own ID relative to its ``Simulation`` and micrograph

# Let's choose 4 random numbers, as our global (Simulation) particle IDs from the first micrograph                            
global_particle_ids = np.random.choice(colliding_micrograph.particles_per_micrograph, 4)

# %%                                                                                                                         
# We can obtain the actual projection images from the ``Simulation``.                                                        
colliding_sim.images[global_particle_ids].show()

# %%                                                                                                                         
# Now, we'll use the ``MicrographSource.get_micrograph()`` method to find the particle's micrograph and ID relative to the micrograph.                                                                                                                   
# We can use these values to find the particle's center.

micrograph_ids, local_particle_ids = np.zeros((4), dtype = int), np.zeros((4), dtype = int)
for i, particle in enumerate(global_particle_ids):
    micrograph_ids[i], local_particle_ids[i] = colliding_micrograph.get_micrograph(particle)

# Print the values and make sure they're all from the first micrograph.
print(f"Micrographs: {list(micrograph_ids)}, Particles: {list(local_particle_ids)}")

# Get the particle centers
centers = np.zeros((4, 2), dtype = int)
for i in range(local_particle_ids.shape[0]):
    centers[i] = colliding_micrograph.centers[micrograph_ids[i]][local_particle_ids[i]]
print(f"Centers: {list(centers)}")

# %%
# Let's use the particle's size and center compare the simulation's images to the actual micrographs.
p_size = colliding_micrograph.particle_box_size
micrograph_squares = np.zeros((4, colliding_micrograph.particle_box_size, colliding_micrograph.particle_box_size))
for i, center in enumerate(centers):
    x, y = center[0], center[1]
    # Calculate the square of the particle
    p_square = colliding_micrograph.clean_micrographs[micrograph_ids[i]].asnumpy()[0][x - p_size//2: x + p_size//2, y - p_size//2: y + p_size//2]
    micrograph_squares[i] = p_square
    
# Plot the squares
Image(micrograph_squares)[:].show()

# %%
# We can see that the particles match up!
# By using the ``MicrographSource.get_particle()`` method, we can find the global (Simulation) ID of the particles.
simulation_ids = np.zeros((4), dtype = int)
for i in range(4):
    simulation_ids[i] = colliding_micrograph.get_particle(micrograph_ids[i], local_particle_ids[i])
print(f"Simulation Particle IDs: {simulation_ids}")
