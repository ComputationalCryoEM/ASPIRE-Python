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
RESOLUTION = 65  # 300 used in paper
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
src.images(0, 10).show()


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

rir = RIRClass2D(src, fspca_basis, fspca_components=100, sample_n=40)
# rir = RIRClass2D(src, fspca_basis, fspca_components=100, sample_n=4000)


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

logger.info("Random Sample:")
random_10 = Image(Orig[np.random.choice(src.n, 10)])
# random_10.show()

logger.info("Classed Sample:")
for c in range(5):
    # this was selecting just the non reflected neighbors and seemed reasonable
    # selection = class_refl[c] == False
    # neighbors = classes[c][selection][:10]  # not refl
    neighbors = classes[c][:10]  # not refl

    neighbors_img = Image(Orig[neighbors])

    # neighbors = classes[c][:10]
    # neighbors_img = Image(Orig[neighbors])
    logger.info("before rot & refl")
    neighbors_img.show()

    co = basis.evaluate_t(neighbors_img)
    logger.info(f"Class {c} after rot/refl")
    # rco = basis.rotate(co, rot[c][:10])
    # rco = basis.rotate(co, rot[c][selection][:10])  # not refl
    rco = basis.rotate(co, rot[c][:10], class_refl[c][:10])

    rotated_neighbors_img = basis.evaluate(rco)
    rotated_neighbors_img.show()
