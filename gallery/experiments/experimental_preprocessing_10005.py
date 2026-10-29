import glob
import os
import re

import numpy as np

from aspire.source import CentersCoordinateSource

PIXEL_SIZE = 1.2156
PRTCL_SIZE = 256
DTYPE = np.float32

# src_flist = list(zip(mrc_flist, box_flist))


root_data_dir = "/Users/gbwright/work/ASPIRE-Python.mrc_src/test_10005"
centers_dir = os.path.join(root_data_dir, "picked_coordinates/TRPV1_coords_35645")

mrc_flist = glob.glob(os.path.join(root_data_dir, "*.mrc"))
centers_flist = glob.glob(os.path.join(centers_dir, "*.dat"))


ctf_result_dir = os.path.join(root_data_dir, "../test_10005_ctf_estimates")
all_ctf_flist = glob.glob(os.path.join(ctf_result_dir, "*.star"))


# re.compile(r"/stack_(\d+)_[^/]*\.mrc$")


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
all_ctf_flist = extract_ids(all_ctf_flist)

ids = sorted(set(centers_flist.keys()) & set(mrc_flist.keys()))
src_flist = []
ctf_flist = []
for i in ids:
    src_flist.append((mrc_flist[i], centers_flist[i]))
    ctf_flist.append(all_ctf_flist[i])


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


src = src.phase_flip().cache()
src = src.downsample(64).cache()
src = src.normalize_background().cache()
src = src.whiten().cache()
src = src.invert_contrast().cache()

src.save("pp.star", overwrite=True, save_mode="single")
