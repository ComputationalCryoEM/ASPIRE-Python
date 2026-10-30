import glob
import os
import re

import numpy as np

from aspire.ctf import estimate_ctf
from aspire.source import CentersCoordinateSource

PIXEL_SIZE = 1.2156
PRTCL_SIZE = 256
DTYPE = np.float32

# src_flist = list(zip(mrc_flist, box_flist))


root_data_dir = "/Users/gbwright/work/ASPIRE-Python.mrc_src/test_10005"
centers_dir = os.path.join(root_data_dir, "picked_coordinates/TRPV1_coords_35645")

mrc_flist = glob.glob(os.path.join(root_data_dir, "*.mrc"))
centers_flist = glob.glob(os.path.join(centers_dir, "*.dat"))

# ---------

ctf_result_dir = os.path.join(root_data_dir, "ctf_estimate_results")

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

ctf_flist = glob.glob(os.path.join(ctf_result_dir, "*.star"))


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
    selected_ctf_flist.append(all_ctf_flist[i])


# ---------

# Setup Source
src = CentersCoordinateSource(
    src_flist,
    pixel_size=PIXEL_SIZE,
    particle_size=PRTCL_SIZE,
    bin_multiplier=3,
    xy_cols=(2, 3),
)

# Assign CTF estimates to src
src.import_aspire_ctf(ctf_flist)


# Legacy preprocessing stack
legacy_src = src.phase_flip().cache()
legacy_src = legacy_src.crop_pad(src.L - 1).cache()
legacy_src = legacy_src.legacy_downsample(64).cache()
legacy_src = legacy_src.legacy_normalize_background().cache()
legacy_src = legacy_src.legacy_whiten().cache()
legacy_src = legacy_src.invert_contrast().cache()
legacy_src.save("legacy_10005_preproc.star", overwrite=True, save_mode="single")

# Development preprocessing stack
dev_src = src.downsample(64).cache()
dev_src = src.phase_flip().cache()
dev_src = dev_src.normalize_background().cache()
dev_src = dev_src.whiten().cache()
dev_src = dev_src.invert_contrast().cache()
dev_src.save("development_10005_preproc.star", overwrite=True, save_mode="single")
