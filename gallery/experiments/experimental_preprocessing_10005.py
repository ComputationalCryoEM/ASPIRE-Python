import glob
import logging
import os
import re

import numpy as np

from aspire.ctf import estimate_ctf
from aspire.denoising import LegacyClassAvgSource
from aspire.reconstruction import MeanEstimator
from aspire.source import CentersCoordinateSource, OrientedSource

logger = logging.getLogger(__name__)

PIXEL_SIZE = 1.2156
PRTCL_SIZE = 256
DTYPE = np.float32

img_size = 129  # Downsample the images/reconstruction to a desired resolution
n_classes = 3000  # How many class averages to compute.
n_nbor = 50  # How many neighbors to stack

# Outputs
# preprocessed_fn = f"10005_preprocessed_{img_size}px.star"
class_avg_fn = f"10005_var_sorted_cls_avgs_m{n_nbor}_{img_size}px.star"
oriented_fn = f"10005_oriented_class_averages_{img_size}px.star"
volume_output_filename = f"10005_abinitio_c{n_classes}_m{n_nbor}_{img_size}px.mrc"


root_data_dir = "/Users/gbwright/work/ASPIRE-Python.mrc_src/test_10005"
centers_dir = os.path.join(root_data_dir, "picked_coordinates/TRPV1_coords_35645")

mrc_flist = glob.glob(os.path.join(root_data_dir, "*.mrc"))
centers_flist = glob.glob(os.path.join(centers_dir, "*.dat"))

# ---------

ctf_result_dir = os.path.join(root_data_dir, "ctf_estimate_results")
ctf_globber = lambda: glob.glob(os.path.join(ctf_result_dir, "*.star"))

# skip re-running estimator if already generated
if not (
    os.path.exists(ctf_result_dir) and len(mrc_flist) == len(ctf_flist := ctf_globber())
):

    estimate_ctf(
        data_folder=root_data_dir,
        pixel_size=PIXEL_SIZE,
        cs=2.7,
        amplitude_contrast=0.07,
        voltage=300,
        num_tapers=4,
        psd_size=1024,
        g_min=30,
        g_max=5,
        output_dir=ctf_result_dir,
        dtype=DTYPE,
        save_ctf_images=True,
        save_noise_images=True,
    )

    ctf_flist = ctf_globber()


# ---------


def extract_ids(paths, regex=r"stack_(\d{4})_"):
    """Return {num_id: path} for every path matching the `regex`."""
    rx = re.compile(regex)

    res = {}
    for path in paths:
        m = rx.search(path)
        if m:
            res[m.group(1)] = path
    return res


# convert into dicts of {id:path...}
mrc_flist = extract_ids(mrc_flist)
centers_flist = extract_ids(centers_flist)
ctf_flist = extract_ids(ctf_flist)

selected_ids = sorted(set(centers_flist.keys()) & set(mrc_flist.keys()))
selected_src_flist = []
selected_ctf_flist = []
for i in selected_ids:
    selected_src_flist.append((mrc_flist[i], centers_flist[i]))
    selected_ctf_flist.append(ctf_flist[i])


# ---------

# Setup Source
src = CentersCoordinateSource(
    selected_src_flist,
    pixel_size=PIXEL_SIZE,
    particle_size=PRTCL_SIZE,
    bin_multiplier=3,
    xy_cols=(2, 3),
)

# Assign CTF estimates to src
src.import_aspire_ctf(selected_ctf_flist)


# Legacy preprocessing stack
legacy_src = src.phase_flip().cache()
legacy_src = legacy_src.crop_pad(src.L - 1).cache()
legacy_src = legacy_src.legacy_downsample(img_size).cache()
legacy_src = legacy_src.legacy_normalize_background().cache()
legacy_src = legacy_src.legacy_whiten().cache()
legacy_src = legacy_src.invert_contrast().cache()
legacy_src.save("legacy_10005_preproc.star", overwrite=True, save_mode="single")

# Development preprocessing stack
dev_src = src.downsample(img_size).cache()
dev_src = src.phase_flip().cache()
dev_src = dev_src.normalize_background().cache()
dev_src = dev_src.whiten().cache()
dev_src = dev_src.invert_contrast().cache()
dev_src.save("development_10005_preproc.star", overwrite=True, save_mode="single")


# ---- Recon

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.
logger.info("Begin Class Averaging")
avgs = LegacyClassAvgSource(legacy_src, n_nbor=n_nbor).cache()

# Save the entire set of class averages to disk so they can be re-used.
avgs.save(class_avg_fn, save_mode="single", overwrite=True)

# We'll continue our pipeline with the first ``n_classes`` from
# ``avgs``.  The classes will be selected by the ``class_selector`` of a
# ``ClassAvgSource``, which in this case will be the class averages
# having the largest variance.  Note global sorting requires computing
# all class averages, which is computationally intensive.
avgs = avgs[:n_classes].cache()


# %%
# Common Line Estimation
# ----------------------
#
# Estimate orientation of projections and assign to source by
# applying ``OrientedSource`` to the class averages from the prior
# step. By default this applies the Common Line with Synchronization
# Voting ``CLSync3N`` method.

logger.info("Begin Orientation Estimation")
oriented_src = OrientedSource(avgs)

# Save off the selected set of class average images, along with the
# estimated orientations and shifts.  These can be reused to
# experiment with alternative volume reconstructions.
oriented_src.save(oriented_fn, save_mode="single", overwrite=True)

# %%
# Volume Reconstruction
# ----------------------
#
# Using the oriented source, attempt to reconstruct a volume.

logger.info("Begin Volume reconstruction")

# Set up an estimator to perform the backprojection.
estimator = MeanEstimator(oriented_src)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
estimated_volume.save(volume_output_filename, overwrite=True)
logger.info(f"Saved Volume to {str(Path(volume_output_filename).resolve())}")
