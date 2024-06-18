import numpy as np
import pytest
from skimage import data
from skimage.transform import radon

from aspire.image import Image

# Image.project and compare results to skimage.radon


def test_image_project():
    image = Image(data.camera().astype(np.float64))
    ny = image.resolution
    angles = np.linspace(0, 360, ny, endpoint=False)
    rads = angles / 180 * np.pi
    s = image.project(rads)

    # add reference skimage radon here

    # compare s with reference
