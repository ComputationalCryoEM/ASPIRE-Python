# flake8: noqa: T001
import logging

import mrcfile
import numpy as np

from aspire.basis import FFBBasis2D, FSPCABasis
from aspire.classification import RIRClass2D
from aspire.image import Image
from aspire.operators import ScalarFilter

# from aspire.source import ArrayImageSource
from aspire.source import Simulation
from aspire.volume import Volume

logger = logging.getLogger(__name__)

##################################################
# Parameters
RESOLUTION = 64  # 300 used in paper
NUMBER_OF_TEST_IMAGES = 4096  # 24000 images
DTYPE = np.float64
##################################################
# Setup

# logger.info("Generates gaussian blob simulation source")
# src = Simulation(
#     n=NUMBER_OF_TEST_IMAGES,
#     L=RESOLUTION,
#     seed=123,
#     dtype=DTYPE,
# )

# # or generate some projections
fh = mrcfile.open("tutorials/data/clean70SRibosome_vol_65p.mrc")
v = Volume(fh.data.astype(DTYPE))
v = v.downsample((RESOLUTION,) * 3)
noise_var = 0  # 0.01 * np.var(np.sum(v[0],axis=0))
noise_filter = ScalarFilter(dim=2, value=noise_var)
src = Simulation(
    L=v.resolution,
    n=NUMBER_OF_TEST_IMAGES,
    vols=v,
    dtype=DTYPE,
    noise_filter=noise_filter,
)
# src.images(0, 10).show()


# ## Trivial rotation for testing invariance
# img = src.images(0,NUMBER_OF_TEST_IMAGES)
# ####img.data = np.transpose(img.data, (0,2,1))
# img.data = img.data[:, ::-1, ::-1] # 180
# img.data = np.rot90(img.data, axes=(1,2)) # 90
# src = ArrayImageSource(img)


# # Peek
# src.images(0,10).show()


logger.info("Setting up FFB")
# Setup a Basis
basis = FFBBasis2D((RESOLUTION, RESOLUTION), dtype=DTYPE)
coefs = basis.evaluate_t(src.images(0, NUMBER_OF_TEST_IMAGES))

logger.info("Setting up FSPCA")
fspca_basis = FSPCABasis(src, basis)
fspca_basis.build(coefs)

# rir = RIRClass2D(src, fspca_basis, fspca_components=100, sample_n=40)
# rir = RIRClass2D(src, fspca_basis, fspca_components=400, sample_n=4000, bispectrum_freq_cutoff=1)
rir = RIRClass2D(
    src,
    fspca_basis,
    fspca_components=100,
    sample_n=40,
    bispectrum_freq_cutoff=3,
)


result = rir.classify()

# debugging/poc
classes, class_refl, rot, corr, _ = result

# print("class_refl")
# print(class_refl.shape)
# print(class_refl[0])

# print("rot")
# print(rot.shape)
# print(rot[0])

# lets peek at first couple image classes:
#   first ten nearest neighbors
Orig = src.images(0, NUMBER_OF_TEST_IMAGES)


def plot_helper(img, refl, columns=5, figsize=(20, 10)):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    for i, im in enumerate(img):
        plt.subplot(img.n_images // columns + 1, columns, i + 1)
        if refl[i]:
            plt.title("Reflected")
        plt.imshow(im, cmap="gray")
    plt.show()


include_refl = False  # I'll have to get some help regarding the reflected set. I don't like the results.

logger.info("Classed Sample:")
for c in range(5):
    # this was selecting just the non reflected neighbors and seemed reasonable
    if include_refl:
        neighbors = classes[c][:10]
    else:
        logger.info("Ignoring Reflected matches")
        selection = class_refl[c] == False
        neighbors = classes[c][selection][:10]  # not refl

    neighbors_img = Image(Orig[neighbors])

    # logger.info("before rot & refl")
    # neighbors_img.show()

    co = basis.evaluate_t(neighbors_img)
    logger.info(f"Class {c} after rot/refl")
    if include_refl:
        rco = basis.rotate(co, rot[c][:10], class_refl[c][:10])
    else:
        # rco = basis.rotate(co, rot[c][:10])
        rco = basis.rotate(co, rot[c][selection][:10])  # not refl

    rotated_neighbors_img = basis.evaluate(rco)
    if include_refl:
        plot_helper(rotated_neighbors_img, class_refl[c][:10])
    else:
        rotated_neighbors_img.show()
