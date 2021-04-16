import logging

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from tqdm import tqdm

from aspire.basis import FBBasis2D, FFBBasis2D, FSPCABasis
from aspire.classification import RIRClass2D
from aspire.image import Image
from aspire.source import ArrayImageSource, Simulation
from aspire.volume import Volume

logger = logging.getLogger(__name__)

##################################################
# Parameters
RESOLUTION = 16  # 300 used in paper
NUMBER_OF_TEST_IMAGES = 3  # 4096  # 24000 images
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
src = Simulation(L=v.resolution, n=NUMBER_OF_TEST_IMAGES, vols=v, dtype=DTYPE)

### Trivial rotation for testing invariance
# img = src.images(0,NUMBER_OF_TEST_IMAGES)
#####img.data = np.transpose(img.data, (0,2,1))
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

rir = RIRClass2D(src, fspca_basis, rank_approx=4000)


rir.classify()
