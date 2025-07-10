import logging
import os.path

import numpy as np
import pytest
from scipy.special import jv

from aspire.basis import Coef, FFBBasis2D
from aspire.nufft import all_backends
from aspire.source import Simulation
from aspire.utils.misc import grid_2d
from aspire.volume import Volume

from ._basis_util import Steerable2DMixin, UniversalBasisMixin, basis_params_2d

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

# Create a test Basis object for each combination of parameters we want to test
test_bases = [FFBBasis2D(L, dtype=dtype) for L, dtype in basis_params_2d]


def show_basis_params(basis):
    # print descriptive test name for parametrized test
    # run pytest with option -rA to see explicitly
    return f"{basis.nres}-{basis.dtype}"


@pytest.mark.parametrize("basis", test_bases, ids=show_basis_params)
class TestFFBBasis2D(Steerable2DMixin, UniversalBasisMixin):
    seed = 9161341

    def _testElement(self, basis, ell, k, sgn):
        ells = basis.angular_indices
        sgns = basis.signs_indices
        ks = basis.radial_indices

        g2d = grid_2d(basis.nres, dtype=basis.dtype)
        mask = g2d["r"] < 1

        r0 = basis.r0[ell][k]

        # TODO: Figure out where these factors of 1 / 2 are coming from.
        # Intuitively, the grid should go from -L / 2 to L / 2, not -L / 2 to
        # L / 4. Furthermore, there's an extra factor of 1 / 2 in the
        # definition of `im` below that may be related.
        r = g2d["r"] * basis.nres / 4

        im = np.zeros((basis.nres, basis.nres), dtype=basis.dtype)
        im[mask] = (
            (-1) ** k
            * np.sqrt(np.pi)
            * r0
            * jv(ell, 2 * np.pi * r[mask])
            / ((2 * np.pi * r[mask]) ** 2 - r0**2)
        )

        if sgn == 1:
            im *= np.sqrt(2) * np.cos(ell * g2d["phi"])
        else:
            im *= np.sqrt(2) * np.sin(ell * g2d["phi"])

        coef_ref = np.zeros(basis.count, dtype=basis.dtype)
        coef_ref[(ells == ell) & (sgns == sgn) & (ks == k)] = 1

        im_ref = Coef(basis, coef_ref).evaluate().asnumpy()[0]

        coef = basis.expand(im)

        # NOTE: These tolerances are expected to be rather loose since the
        # above expression for `im` is derived from the analytical formulation
        # (eq. 6 in Zhao and Singer, 2013) and does not take into account
        assert np.allclose(im, im_ref, atol=1e-1)
        assert np.allclose(coef, coef_ref, atol=1e-1)

    def testElements(self, basis):
        ells = [1, 1, 1, 1]
        ks = [1, 2, 1, 2]
        sgns = [-1, -1, 1, 1]

        for ell, k, sgn in zip(ells, ks, sgns):
            self._testElement(basis, ell, k, sgn)

    def testShift(self, basis):
        """
        Compare shifting using Image with shifting provided by the Basis.

        Note the Basis shift method converts from FB to Image space and back.
        """

        n_img = 3
        test_shift = np.array([10, 2])

        # Construct some synthetic data
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                basis.dtype
            ),
            pixel_size=1.234,
        ).downsample(basis.nres)

        src = Simulation(L=basis.nres, n=n_img, vols=v, dtype=basis.dtype)

        # Shift images using the Image method directly
        shifted_imgs = src.images[:n_img].shift(test_shift)

        # Convert original images to basis coefficients
        f_imgs = basis.evaluate_t(src.images[:n_img])

        # Use the basis shift method
        f_shifted_imgs = basis.shift(f_imgs, test_shift)

        # Compute diff between the shifted image sets
        diff = shifted_imgs.asnumpy() - basis.evaluate(f_shifted_imgs).asnumpy()

        # Compute mask to compare only the core of the shifted images
        g = grid_2d(basis.nres, indexing="yx", normalized=False)
        mask = g["r"] > basis.nres / 2
        # Masking values outside radius to 0
        diff = np.where(mask, 0, diff)

        # Compute and check error
        rmse = np.sqrt(np.mean(np.square(diff), axis=(1, 2)))
        logger.info(f"RMSE shifted image diffs {rmse}")
        assert np.allclose(rmse, 0, atol=1e-5)

        # Check pixel_size passthrough
        np.testing.assert_array_equal(f_imgs.pixel_size, f_shifted_imgs.pixel_size)


params = [pytest.param(512, np.float32, marks=pytest.mark.expensive)]


@pytest.mark.skipif(
    all_backends()[0] == "cufinufft", reason="Not enough memory to run via GPU"
)
@pytest.mark.parametrize(
    "L, dtype",
    params,
)
def testHighResFFBBasis2D(L, dtype):
    seed = 42
    basis = FFBBasis2D(L, dtype=dtype)
    sim = Simulation(
        n=1,
        L=L,
        C=1,
        dtype=dtype,
        amplitudes=1,
        offsets=0,
        seed=seed,
    )
    im = sim.images[0]

    # Round trip
    coef = basis.evaluate_t(im)
    im_ffb = basis.evaluate(coef)

    # Mask to compare inside disk of radius 1.
    mask = grid_2d(L, normalized=True)["r"] < 1
    np.testing.assert_allclose(
        im_ffb.asnumpy()[0][mask], im.asnumpy()[0][mask], rtol=1e-05, atol=1e-4
    )
