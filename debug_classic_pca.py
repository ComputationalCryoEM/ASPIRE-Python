import logging

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from tqdm import tqdm

# from aspire.classification import RIRClass2D
from aspire.basis import FBBasis2D, FFBBasis2D, FSPCABasis
from aspire.image import Image
from aspire.source import ArrayImageSource, Simulation
from aspire.volume import Volume

logger = logging.getLogger(__name__)

##################################################
# Parameters
RESOLUTION = 64
NUMBER_OF_TEST_IMAGES = 4096
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

# or generate some projections
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


# Orig is n_images*res*res, but we want (resolution*resolution) rows by n_images (cols)
Orig = src.images(0, src.n).asnumpy()
A = np.moveaxis(Orig, 0, -1).reshape(RESOLUTION * RESOLUTION, NUMBER_OF_TEST_IMAGES)
# plt.imshow(A[:,3].reshape(RESOLUTION, RESOLUTION));plt.show()  # peek

# demean
A_demean = A - np.mean(A, axis=0)

logger.info("SVD")
u, s, vt = np.linalg.svd(A_demean, full_matrices=False)

print(f"u.shape {u.shape}")
plt.semilogy(s)
plt.show()

eigenimages = np.moveaxis(
    u.reshape(RESOLUTION, RESOLUTION, NUMBER_OF_TEST_IMAGES), -1, 0
)

# for k in range(10):
#     plt.imshow(eigenimages[k])
#     plt.show()
eigenplot = np.empty((10, RESOLUTION, 10, RESOLUTION))
for k in range(10):
    for j in range(10):
        eigenplot[k, :, j, :] = eigenimages[k * 10 + j]
eigenplot = eigenplot.reshape(10 * RESOLUTION, 10 * RESOLUTION)
plt.imshow(eigenplot)
plt.show()


k = 100
u_est = u[:, :k]
print(f"u_est: {u_est.shape}")
A_compressed = u_est.T @ A
print(f"A_compressed: {A_compressed.shape}")
A_est = u_est @ A_compressed

A_img = np.moveaxis(A_est.reshape(RESOLUTION, RESOLUTION, NUMBER_OF_TEST_IMAGES), -1, 0)

# for k in range(10):
#     plt.imshow(A_img[k])
#     plt.show()
