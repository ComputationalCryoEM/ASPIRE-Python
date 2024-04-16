"""
Notes on env setup (Linux x86_64)

conda create -n cryojax-aspire-env -c conda-forge python=3.11
conda activate cryojax-aspire-env
python -m pip install cryojax
# note if installing from source, will need `git-lfs pull` to download data files.
# note that we actually ended up using the `grid-search` development branch (not the pypi package above)
# Todo, update these notes when polishing the example or wait for their next release to publish...


git clone -b develop git@github.com:ComputationalCryoEM/ASPIRE-Python
cd ASPIRE-Python
pip install -e ".[dev]"
# Installing from pypi was similar, no issues.  For publishing post we should just use the pypi release...

"""

# CryoJAX imports

from functools import partial

import cryojax as cx
import cryojax.simulator as cxs
import equinox as eqx
import equinox.internal as eqxi

# Jax imports
import jax
import jax.numpy as jnp
import numpy as np
from cryojax.data import read_array_with_spacing_from_mrc
from cryojax.image.operators import FourierExp2D
from cryojax.rotations import SO3
from jaxtyping import PRNGKeyArray, PyTree

# Plotting imports and functions
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# First, load the scattering potential and projection method
filename = "./data/ribosome_4ug0_scattering_potential_from_cistem.mrc"
real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(filename)
potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
    real_voxel_grid, voxel_size, pad_scale=2
)
integrator = cxs.FourierSliceExtract(interpolation_order=1)

# Initialize the instrument
voltage_in_kilovolts = 300.0
dose = cxs.ElectronDose(electrons_per_angstrom_squared=100.0)

optics = cxs.WeakPhaseOptics(
    ctf=cxs.CTF(
        defocus_u_in_angstroms=10000.0,
        defocus_v_in_angstroms=9000.0,
        astigmatism_angle=20.0,
        amplitude_contrast_ratio=0.07,
    )
)

detector = cxs.PoissonDetector(dqe=cxs.IdealDQE(fraction_detected_electrons=1.0))

instrument_with_detector = cxs.Instrument(
    voltage_in_kilovolts, dose=dose, optics=optics, detector=detector
)


# TODO: Figure out how to reduce the noise to something easier to run a demo sized recon with...
# Then, choose a model for the solvent. The amplitude is the
# (squared) characteristic phase shift of the ice phase shifts, and the length_scale is
# their characteristic length scale.
solvent = cxs.GaussianIce(
    variance=FourierExp2D(amplitude=0.005**2, length_scale=2.0 * potential.voxel_size)
)

# ... and finally the config
shape = (89, 89)
pixel_size = potential.voxel_size  # Angstroms
image_size = np.asarray(shape) * pixel_size
config = cxs.ImageConfig(shape, pixel_size, pad_scale=1.1)


@partial(eqx.filter_vmap, in_axes=(0, None), out_axes=eqxi.if_mapped(axis=0))
def make_pipeline(key: PRNGKeyArray, no_vmap: tuple[PyTree, ...]) -> cxs.ImagePipeline:
    config, potential, integrator, instrument, solvent = no_vmap
    # ... instantiate rotations
    rotation = SO3.sample_uniform(key)
    # ... now in-plane translation
    ny, nx = config.shape

    # TODO, remove this
    in_plane_offset_in_angstroms = (
        jax.random.uniform(key, (2,), minval=-0.45, maxval=0.45)
        * jnp.asarray((nx, ny))
        * config.pixel_size
        * 0  # hack
    )
    # ... convert 2D in-plane translation to 3D, setting the out-of-plane translation to
    # zero
    offset_in_angstroms = jnp.pad(in_plane_offset_in_angstroms, ((0, 1),))
    # ... build the pose
    pose = cxs.QuaternionPose.from_rotation_and_translation(
        rotation, offset_in_angstroms
    )

    # # TODO: Ask what I'm doing wrong here ...
    # pose = cxs.QuaternionPose.from_rotation(rotation)

    # ... build the Specimen and ImagePipeline as usual and return
    specimen = cxs.Specimen(potential, integrator, pose)
    return cxs.ImagePipeline(config, specimen, instrument, solvent=solvent)


# Generate RNG keys
number_of_poses = 5000
keys = jax.random.split(jax.random.PRNGKey(12345), number_of_poses)

# ... instantiate the pipeline
pipeline = make_pipeline(
    keys, (config, potential, integrator, instrument_with_detector, solvent)
)


# ... specify which leaves we would like to vmap over
where = lambda pipeline: pipeline.specimen.pose
# ... use a cryojax wrapper to return a filter_spec
filter_spec = cx.get_filter_spec(pipeline, where)


@partial(cx.filter_vmap_with_spec, filter_spec=filter_spec)
def compute_image_stack(pipeline, key):
    """Compute a batch of images at different poses,
    specified by the `filter_spec`.
    """
    # key = jax.random.PRNGKey(77)
    # image = pipeline.render()
    image = pipeline.sample(key)
    return image - image.mean()


res = compute_image_stack(pipeline, keys)
# convert to numpy array
res_np = np.array(res).astype(np.float64)
# save off
np.save(f"res_{number_of_poses}_px{shape[0]}.npy", res)

#### Begin ASPIRE

import aspire

# Load into ASPIRE
src = aspire.source.ArrayImageSource(res_np)
src.images[:10].show(colorbar=False)

# Run some preprocessing methods
src = src.normalize_background().whiten()
src.images[:10].show(colorbar=False)

# Class averaging


# set parameters
n_classes = 200
n_nbor = 10

# We will customize our class averaging source. Note that the
# ``fspca_components`` and ``bispectrum_components`` were selected for
# this small tutorial.
rir = aspire.classification.RIRClass2D(
    src,
    fspca_components=40,
    bispectrum_components=30,
    n_nbor=n_nbor,
)


avgs = aspire.denoising.DebugClassAvgSource(
    src=src,
    classifier=rir,
)

# We'll continue our pipeline with the first ``n_classes`` from ``avgs``.
avgs = avgs[:n_classes]
avgs.images[:].save("avgs.mrcs", overwrite=True)

# Show class averages
avgs.images[0:10].show()

# Orientation Estimation
# Instantiate an ``OrientedSource``.
oriented_src = aspire.source.OrientedSource(avgs)


# Setup an estimator to perform the back projection.
estimator = aspire.reconstruction.MeanEstimator(oriented_src)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()

estimated_volume.save("jaxvol.mrc", overwrite=True)
