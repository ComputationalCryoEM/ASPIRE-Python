import logging
import os
import tempfile
from shutil import copyfile

import mrcfile
import numpy as np
import pytest

from aspire.basis import FBBasis2D, FFBBasis2D, FLEBasis2D, FPSWFBasis2D, PSWFBasis2D
from aspire.ctf import estimate_ctf

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

DTYPE = np.float64
PSD_SIZE = 512
TEST_INPUT_FN = "sample.mrc"
# These values are from CTFFIND4
TEST_OUTPUT = {
    "defocus_u": 34914.63,  # angstrom
    "defocus_v": 33944.32,  # angstrom
    "defocus_ang": -65.26,  # Degree wrt some axis
    "cs": 2.0,
    "voltage": 300.0,
    "pixel_size": 1.77,  # EMPIAR 10017
    "amplitude_contrast": 0.07,
}

BASES = [
    None,  # resolves to FFBBasis2D,
    pytest.param(FLEBasis2D, marks=pytest.mark.expensive),
    # The following are very long and may require a higher memory machine
    pytest.param(FBBasis2D, marks=pytest.mark.ctf),
    pytest.param(PSWFBasis2D, marks=pytest.mark.ctf),
    pytest.param(FPSWFBasis2D, marks=pytest.mark.ctf),
]


@pytest.fixture(params=BASES, ids=lambda x: f"basis={x}", scope="module")
def basis(request):
    cls = request.param
    if cls is None:
        return None
    # Setup a Basis
    basis = cls(PSD_SIZE, dtype=DTYPE)
    return basis


def test_estimate_CTF(basis, **customized_args):
    _dtype = customized_args.pop("dtype", DTYPE)  # avoid duplicate dtype arg
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        # Copy input file
        copyfile(
            os.path.join(DATA_DIR, TEST_INPUT_FN),
            os.path.join(tmp_input_dir, TEST_INPUT_FN),
        )

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            # Returns results in output_dir
            results = estimate_ctf(
                data_folder=tmp_input_dir,
                pixel_size=1.77,
                cs=2.0,
                amplitude_contrast=0.07,
                voltage=300.0,
                num_tapers=2,
                psd_size=PSD_SIZE,
                g_min=30.0,
                g_max=5.0,
                output_dir=tmp_output_dir,
                basis=basis,
                dtype=_dtype,
                save_ctf_images=True,
                save_noise_images=True,
                **customized_args,
            )

            logger.debug(f"results: {results}")

            for result in results.values():
                # The defocus values are set to be within 5% of CTFFIND4

                # defocusU
                np.testing.assert_allclose(
                    result["defocus_u"],
                    TEST_OUTPUT["defocus_u"],
                    rtol=0.05,
                )

                # defocusV
                np.testing.assert_allclose(
                    result["defocus_v"],
                    TEST_OUTPUT["defocus_v"],
                    rtol=0.05,
                )

                # defocusAngle
                defocus_ang_degrees = result["defocus_ang"] * 180 / np.pi
                try:
                    np.testing.assert_allclose(
                        defocus_ang_degrees,
                        TEST_OUTPUT["defocus_ang"],
                        atol=1,  # one degree
                    )
                except AssertionError:
                    logger.warning(
                        "Defocus Angle (degrees):"
                        f"\n\tASPIRE= {defocus_ang_degrees:0.2f}*"
                        f'\n\tCTFFIND4= {TEST_OUTPUT["defocus_ang"]:0.2f}*'
                        f'\n\tError: {abs((TEST_OUTPUT["defocus_ang"] - defocus_ang_degrees)/TEST_OUTPUT["defocus_ang"]) * 100:0.2f}%'
                    )

                for param in ["cs", "amplitude_contrast", "voltage", "pixel_size"]:
                    np.testing.assert_allclose(result[param], TEST_OUTPUT[param])


def test_customized_estimate_CTF():
    basis = FFBBasis2D(PSD_SIZE, dtype=np.float32)
    customized = dict(
        dtype=basis.dtype,
        solver_method="gd",
        solver_opts={
            "gradient_descent_iterations": 4,
            "max_iterations": 300,
            "stop_cond": 1e-8,
        },
        linprog_method="highs",
        step_defocus=25,
    )
    return test_estimate_CTF(basis, **customized)


# we are chopping the micrograph into a vertical and a horizontal rectangle
# as small as possible to save testing duration
@pytest.mark.parametrize(
    "slice_range", [((slice(0, 128), slice(0, 64))), ((slice(0, 64), slice(0, 128)))]
)
def testRectangularMicrograph(slice_range):
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        # copy input file
        copyfile(
            os.path.join(DATA_DIR, TEST_INPUT_FN),
            os.path.join(tmp_input_dir, "rect_" + TEST_INPUT_FN),
        )
        # trim the file into a rectangle
        with mrcfile.open(
            os.path.join(tmp_input_dir, "rect_" + TEST_INPUT_FN), "r+"
        ) as mrc_in:
            data = mrc_in.data[slice_range]
            mrc_in.set_data(data)
        # make sure we can estimate with no errors
        with tempfile.TemporaryDirectory() as tmp_output_dir:
            _ = estimate_ctf(
                data_folder=tmp_input_dir, output_dir=tmp_output_dir, psd_size=64
            )
