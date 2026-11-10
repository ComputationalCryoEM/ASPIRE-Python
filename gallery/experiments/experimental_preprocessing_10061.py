"""
Preprocessing Pipeline - Experimental Data EMPIAR 10061
=======================================================

This notebook introduces a selection of components corresponding to
using a `CentersCoordinateSource` in conjunction with CTF estimation
for real cryo-EM data.

Specifically this pipeline uses the EMPIAR 10061
micrographs and EMAN coordinates available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10061
"""

# %%
# Imports
# -------
# Import packages that will be used throughout this experiment.

import glob
import logging
import os
import re
from pathlib import Path

import numpy as np

from aspire.ctf import estimate_ctf
from aspire.source import BoxesCoordinateSource

logger = logging.getLogger(__name__)

# %%
# Parameters
# ---------------
#

# These parameters are taken from the published data.
PIXEL_SIZE = 0.3185

# Data location
root_data_dir = "10061/data"
# Build list of input files.
mrc_flist = glob.glob(os.path.join(root_data_dir, "**/*.mrc"))
centers_flist = glob.glob(os.path.join(root_data_dir, "*.box"))


# These are user configurable
DTYPE = np.float64
IMG_SIZE = 129  # Desired resolution for downsampled images


# %%
# CTF Estimation
# --------------
# This section demonstrates configuration and execution of CTF estimation using ASPIRE.
# The results are saved, and imported in a later stage.

ctf_result_dir = "./ctf_estimate_results_10061_normpatched"
ctf_globber = lambda: glob.glob(os.path.join(ctf_result_dir, "**/*.star"))

# Because CTF estimation may take some time, This code will skip
# re-running estimator if results already exist in the expected
# location and have the expected length.
if not (
    os.path.exists(ctf_result_dir) and len(mrc_flist) == len(ctf_flist := ctf_globber())
):
    # Most of these are default values, shown here for
    # convenience/reference, except `cs`, `amplitude_contrast`, and
    # `voltage`, which should correspond to the published dataset.
    estimate_ctf(
        data_folder=root_data_dir,
        pixel_size=PIXEL_SIZE,
        cs=2.7,
        amplitude_contrast=0.1,
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

    # Collect the resulting set of CTF estimation results.
    ctf_flist = ctf_globber()


# %%
# Map Source Paths
# ----------------
# The following code handles organizing and mapping the MRC, centers,
# and CTF estimates into coherent lists for the
# `BoxesCoordinateSource` to consume.  `CoordinateSource` classes
# generally expect an implicitly organized mapping of the input
# micrographs.  This code achieves that by using a helper function to
# parse out a unique and sortable identifier from the filenames.  This
# type of code may need to be customized for other datasets.


def extract_ids(paths, regex=r"EMD-(.*)\."):
    """Return `{num_id: path,}` dict for every path matching the `regex`."""
    rx = re.compile(regex)

    res = {}
    for path in paths:
        m = rx.search(path)
        if m:
            res[m.group(1)] = path
    return res


# Convert each type of available data into dicts of {id:path...}
mrc_flist = extract_ids(mrc_flist)
centers_flist = extract_ids(centers_flist)
ctf_flist = extract_ids(ctf_flist)

# Define the set of ids where both the MRC and centers are provided.
# This handles the situation where no particles were chosen from some micrographs.
selected_ids = sorted(set(centers_flist.keys()) & set(mrc_flist.keys()))

# Loop over the `selected_ids` generating sorted lists of MRC, centers, and CTF input files.
selected_src_flist = []
selected_ctf_flist = []
for i in selected_ids:
    selected_src_flist.append((mrc_flist[i], centers_flist[i]))
    selected_ctf_flist.append(ctf_flist[i])

# %%
# Setup BoxesCoordinateSource
# -----------------------------
#

# Setup Source
src = BoxesCoordinateSource(
    selected_src_flist,
    pixel_size=PIXEL_SIZE,
).cache()

# Assign CTF estimates to src
# This file list order must match the src list in `BoxesCoordinateSource`
src.import_aspire_ctf(selected_ctf_flist)

# %%
# Preprocessing
# -------------
# Execute the preprocessing stack, saving results

# Example preprocessing stack
src = src.phase_flip().cache()
src = src.crop_pad(src.L - 1).cache()
src = src.legacy_downsample(IMG_SIZE).cache()
src = src.legacy_normalize_background().cache()
src = src.legacy_whiten().cache()
src = src.invert_contrast().cache()
src.save(
    f"10061_legacy_preproc_{IMG_SIZE}.star", overwrite=True, save_mode="single"
)
