import logging
import os
import tempfile
import warnings
from itertools import product

import numpy as np
import pytest
from numpy import pi
from pytest import raises, skip

from aspire.source import _LegacySimulation
from aspire.utils import Rotation, anorm, grid_2d, powerset, utest_tolerance
from aspire.volume import (
    AsymmetricVolume,
    CnSymmetricVolume,
    CnSymmetryGroup,
    SymmetryGroup,
    TSymmetryGroup,
    Volume,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)


def res_id(params):
    return f"res={params}"


RES = [42, 43]
TEST_PX_SZ = 4.56


@pytest.fixture(params=RES, ids=res_id, scope="module")
def res(request):
    return request.param


def dtype_id(params):
    return f"dtype={params}"


DTYPES = [np.float32, np.float64]


@pytest.fixture(params=DTYPES, ids=dtype_id, scope="module")
def dtype(request):
    return request.param


N = 3


# Note, range shifted by one to avoid zero division errors.
@pytest.fixture
def data_1(res, dtype):
    return np.arange(1, 1 + N * res**3, dtype=dtype).reshape(N, res, res, res)


@pytest.fixture
def data_2(data_1):
    return 123 * data_1.copy()


@pytest.fixture
def data_12(data_1, data_2):
    return np.concatenate([data_1, data_2], axis=0).reshape(2, *data_1.shape)


@pytest.fixture
def vols_1(data_1):
    return Volume(data_1)


@pytest.fixture
def vols_2(data_2):
    return Volume(data_2, pixel_size=TEST_PX_SZ)


@pytest.fixture
def vols_12(data_12):
    return Volume(data_12)


@pytest.fixture
def asym_vols(res, dtype):
    vols = AsymmetricVolume(L=res, C=N, dtype=dtype, seed=0).generate()
    return vols


@pytest.fixture(scope="module")
def symmetric_vols(res, dtype):
    vol_c3 = CnSymmetricVolume(L=res, C=1, order=3, dtype=dtype, seed=0).generate()
    vol_c4 = CnSymmetricVolume(L=res, C=1, order=4, dtype=dtype, seed=0).generate()
    return vol_c3, vol_c4


@pytest.fixture(scope="module")
def vols_hot_cold(res, dtype):
    L = res
    n_vols = 5

    # Generate random locations for hot/cold spots, each at a distance of approximately
    # L // 4 from (0, 0, 0). Note, these points are considered to be in (z, y, x) order.
    hot_cold_locs = np.random.uniform(low=-1, high=1, size=(n_vols, 2, 3))
    hot_cold_locs = np.round(
        (hot_cold_locs / np.linalg.norm(hot_cold_locs, axis=-1)[:, :, None]) * (L // 4)
    ).astype("int")

    # Generate Volumes, each with one hot and one cold spot.
    vols = np.zeros((n_vols, L, L, L), dtype=dtype)
    vol_center = np.array((L // 2, L // 2, L // 2), dtype="int")
    for i in range(n_vols):
        vols[i][tuple(vol_center + hot_cold_locs[i, 0])] = 1
        vols[i][tuple(vol_center + hot_cold_locs[i, 1])] = -1
    vols = Volume(vols)

    return vols, hot_cold_locs, vol_center


@pytest.fixture
def random_data(res, dtype):
    return np.random.randn(res, res, res).astype(dtype)


@pytest.fixture
def vec(data_1, res):
    return data_1.reshape(N, res**3)


def test_repr(vols_12):
    r = repr(vols_12)
    logger.info(f"Volume repr:\n{r}")


def test_noncube(dtype):
    """Test that an irregular Volume array raises."""
    with raises(ValueError, match=r".* cubed .*"):
        _ = Volume(np.empty((4, 5, 6), dtype=dtype))


def test_asnumpy(data_1, vols_1):
    assert np.all(data_1 == vols_1.asnumpy())


def test_astype(vols_1, dtype):
    if dtype == np.float64:
        new_dtype = np.float32
    elif dtype == np.float32:
        new_dtype = np.float64
    else:
        skip("Skip numerically comparing non float types.")

    v2 = vols_1.astype(new_dtype)
    assert isinstance(v2, Volume)
    assert np.allclose(v2.asnumpy(), vols_1.asnumpy())
    assert v2.dtype == new_dtype


def test_astype_copy(vols_1):
    """
    `astype(copy=False)` is an optimization partially mimicked from numpy.
    """
    # Same dtype, copy=False
    v2 = vols_1.astype(vols_1.dtype, copy=False)
    # Details should match,
    assert isinstance(v2, Volume)
    assert np.allclose(v2.asnumpy(), vols_1.asnumpy())
    assert v2.dtype == vols_1.dtype
    # and they should share the same memory (np.ndarray.base).
    assert v2.asnumpy().base is vols_1.asnumpy().base

    # Same dtype, default copy=True
    v2 = vols_1.astype(vols_1.dtype)
    # Details should match,
    assert isinstance(v2, Volume)
    assert np.allclose(v2.asnumpy(), vols_1.asnumpy())
    assert v2.dtype == vols_1.dtype
    # but they should not share the same memory (np.ndarray.base)
    assert v2.asnumpy().base is not vols_1.asnumpy().base


def test_getter(vols_1, data_1):
    k = np.random.randint(N)
    assert np.all(vols_1[k] == data_1[k])


def test_setter(vols_1, random_data):
    k = np.random.randint(N)
    ref = vols_1.asnumpy().copy()
    # Set one entry in the stack with new data
    vols_1[k] = random_data

    # Assert we have updated the kth volume
    assert np.allclose(vols_1[k], random_data)

    # Assert the other volumes are not updated.
    inds = np.arange(N) != k
    assert np.all(vols_1[inds] == ref[inds])


def testLen(vols_1, random_data):
    assert len(vols_1) == N

    # Also test a single volume
    assert len(Volume(random_data)) == 1


def test_add(vols_1, vols_2, data_1, data_2):
    result = vols_1 + vols_2
    assert np.all(result == data_1 + data_2)
    assert isinstance(result, Volume)


def test_scalar_add(vols_1, data_1):
    result = vols_1 + 42
    assert np.all(result == data_1 + 42)
    assert isinstance(result, Volume)


def test_scalar_r_add(vols_1, data_1):
    result = 42 + vols_1
    assert np.all(result == data_1 + 42)
    isinstance(result, Volume)


def test_sub(vols_1, vols_2, data_1, data_2):
    result = vols_1 - vols_2
    assert np.all(result == data_1 - data_2)
    assert isinstance(result, Volume)


def test_scalar_sub(vols_1, data_1):
    result = vols_1 - 42
    np.all(result == data_1 - 42)
    assert isinstance(result, Volume)


def test_scalar_r_sub(vols_1, data_1):
    result = 42 - vols_1
    assert np.all(result == 42 - data_1)
    assert isinstance(result, Volume)


def test_scalar_mul(vols_1, data_2):
    result = vols_1 * 123
    assert np.all(result == data_2)
    assert isinstance(result, Volume)


def test_scalar_r_mul(vols_1, data_2):
    result = 123 * vols_1
    assert np.all(result == data_2)
    assert isinstance(result, Volume)


def test_scalar_div(vols_2, vols_1):
    result = vols_2 / 123
    assert np.allclose(result, vols_1)


def test_right_scalar_div(vols_2, data_1):
    result = 123 / vols_2
    assert np.allclose(result, 1 / data_1)


def test_div(vols_2, vols_1):
    result = vols_2 / vols_1
    assert np.allclose(result, 123)


def test_right_div(data_2, vols_1):
    result = data_2 / vols_1
    np.allclose(result, 123)


def test_save_load(vols_1):
    # Create a tmpdir in a context. It will be cleaned up on exit.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the Volume object into an MRC files
        mrcs_filepath = os.path.join(tmpdir, "test.mrc")
        vols_1.save(mrcs_filepath)

        # Load saved MRC file as a Volume of dtypes single and double.
        vols_loaded_single = Volume.load(mrcs_filepath, dtype=np.float32)
        vols_loaded_double = Volume.load(mrcs_filepath, dtype=np.float64)

        # Check that loaded data are Volume instances and compare to original volume.
        assert isinstance(vols_loaded_single, Volume)
        assert np.allclose(vols_1, vols_loaded_single)
        assert isinstance(vols_loaded_double, Volume)
        assert np.allclose(vols_1, vols_loaded_double)
        assert vols_loaded_single.pixel_size is None, "Pixel size should be None"
        assert vols_loaded_double.pixel_size is None, "Pixel size should be None"


def test_save_overwrite(caplog):
    """
    Test that the overwrite flag behaves as expected.
    - overwrite=True: Overwrites the existing file.
    - overwrite=False: Raises an error if the file exists.
    - overwrite=None: Renames the existing file and saves the new one.
    """
    # Create a tmp dir for this test output
    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        mrc_path = os.path.join(tmpdir_name, "og.mrc")
        base, ext = os.path.splitext(mrc_path)

        # Create and save the first image
        vol1 = Volume(np.ones((1, 8, 8, 8), dtype=np.float32))
        vol1.save(mrc_path, overwrite=True)

        # Case 1: overwrite=True (should overwrite the existing file)
        vol2 = Volume(2 * np.ones((1, 8, 8, 8), dtype=np.float32))
        vol2.save(mrc_path, overwrite=True)

        # Load and check if vol2 has overwritten vol1
        vol2_loaded = Volume.load(mrc_path)
        np.testing.assert_allclose(vol2.asnumpy(), vol2_loaded.asnumpy())

        # Case 2: overwrite=False (should raise an overwrite error)
        vol3 = Volume(3 * np.ones((1, 8, 8, 8), dtype=np.float32))

        with pytest.raises(
            ValueError,
            match="File '.*' already exists; set overwrite=True to overwrite it",
        ):
            vol3.save(mrc_path, overwrite=False)

        # Case 3: overwrite=None (should rename the existing file and save vol3 with original filename)
        with caplog.at_level(logging.INFO):
            vol3.save(mrc_path, overwrite=None)

            # Check that the existing file was renamed and logged
            assert f"Found existing file with name {mrc_path}" in caplog.text

            # Find the renamed file by checking the directory contents
            renamed_file = None
            for filename in os.listdir(tmpdir_name):
                if filename.startswith("og_") and filename.endswith(".mrc"):
                    renamed_file = os.path.join(tmpdir_name, filename)
                    break

            assert renamed_file is not None, "Renamed file not found"

        # Load and check that vol3 was saved to the original path
        vol3_loaded = Volume.load(mrc_path)
        np.testing.assert_allclose(vol3.asnumpy(), vol3_loaded.asnumpy())

        # Also check that the renamed file still contains vol2's data
        vol2_loaded_renamed = Volume.load(renamed_file)
        np.testing.assert_allclose(vol2.asnumpy(), vol2_loaded_renamed.asnumpy())


def test_volume_pixel_size(vols_2):
    """
    Test volume is storing pixel_size attribute.
    """
    assert np.isclose(TEST_PX_SZ, vols_2.pixel_size), "Incorrect Volume pixel_size"


def test_save_load_pixel_size(vols_2):
    # Create a tmpdir in a context. It will be cleaned up on exit.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the Volume object into an MRC files
        mrcs_filepath = os.path.join(tmpdir, "test.mrc")
        vols_2.save(mrcs_filepath)

        # Load saved MRC file as a Volume of dtypes single and double.
        vols_loaded_single = Volume.load(mrcs_filepath, dtype=np.float32)
        vols_loaded_double = Volume.load(mrcs_filepath, dtype=np.float64)

        # Confirm the pixel size is loaded
        np.testing.assert_approx_equal(
            vols_loaded_single.pixel_size,
            vols_2.pixel_size,
            err_msg="Incorrect pixel size in singles.",
        )
        np.testing.assert_approx_equal(
            vols_loaded_double.pixel_size,
            vols_2.pixel_size,
            err_msg="Incorrect pixel size in doubles.",
        )


def test_project(vols_hot_cold):
    """
    We project Volumes containing random hot/cold spots using random rotations and check that
    hot/cold spots in the projections are in the expected locations.
    """
    vols, hot_cold_locs, vol_center = vols_hot_cold
    dtype = vols.dtype
    L = vols.resolution

    # Generate random rotations.
    rots = Rotation.generate_random_rotations(n=vols.n_vols, dtype=dtype)

    # To find the expected location of hot/cold spots in the projections we rotate the 3D
    # vector of locations by the transpose, ie. rots.invert(), (since our projections are
    # produced by rotating the underlying grid) and then project along the z-axis.

    # Expected location of hot/cold spots relative to (0, 0, 0) origin in (x, y, z) order.
    # Note, we write the simpler `(x, y, z) @ rots` in place of `(rots.T @ (x, y, z).T).T`
    expected_hot_cold = hot_cold_locs[..., ::-1] @ rots.matrices

    # Expected location of hot/cold spots relative to center (L/2, L/2, L/2) in (z, y, x) order.
    # Then projected along z-axis by dropping the z component.
    expected_locs = np.round(expected_hot_cold[..., ::-1] + vol_center)[..., 1:]

    # Generate projection images.
    projections = vols.project(rots)

    # Check that new hot/cold spots are within 1 pixel of expected locations.
    for i in range(vols.n_vols):
        p = projections.asnumpy()[i]
        new_hot_loc = np.unravel_index(np.argmax(p), (L, L))
        new_cold_loc = np.unravel_index(np.argmin(p), (L, L))
        np.testing.assert_allclose(new_hot_loc, expected_locs[i, 0], atol=1)
        np.testing.assert_allclose(new_cold_loc, expected_locs[i, 1], atol=1)


def test_project_axes(vols_1, dtype):
    L = vols_1.resolution
    # first test with synthetic data
    # Create a stack of rotations to test.
    r_stack = np.empty((12, 3, 3), dtype=dtype)
    for r, ax in enumerate(["x", "y", "z"]):
        r_stack[r] = Rotation.about_axis(ax, 0).matrices
        # We'll consider the multiples of pi/2.
        r_stack[r + 3] = Rotation.about_axis(ax, pi / 2).matrices
        r_stack[r + 6] = Rotation.about_axis(ax, pi).matrices
        r_stack[r + 9] = Rotation.about_axis(ax, 3 * pi / 2).matrices

    # Project a Volume with all the test rotations
    vol_id = 1  # select a volume from Volume stack
    img_stack = vols_1[vol_id].project(r_stack)

    for r in range(len(r_stack)):
        # Get result of test projection at center of Image.
        prj_along_axis = img_stack.asnumpy()[r][L // 2, L // 2]

        # For Volume, take mean along the axis of rotation.
        vol_along_axis = np.mean(vols_1.asnumpy()[vol_id], axis=r % 3)
        # If volume is centered, take middle value, else take the mean of a 2x2 window.
        if L % 2 == 1:
            vol_along_axis = vol_along_axis[L // 2, L // 2]
        else:
            vol_along_axis = np.mean(
                vol_along_axis[L // 2 - 1 : L // 2 + 1, L // 2 - 1 : L // 2 + 1]
            )
        # The projection and Volume should be equivalent
        #  centered along the rotation axis for multiples of pi/2.
        assert np.allclose(vol_along_axis, prj_along_axis)

    # test with saved ribosome data
    results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_down8_imgs32.npy"))
    vols = Volume(np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_down8.npy")))
    rots = np.load(os.path.join(DATA_DIR, "rand_rot_matrices32.npy"))
    rots = np.moveaxis(rots, 2, 0)

    # Note, transforming rotations to compensate for legacy grid convention used in saved data.
    rots = _LegacySimulation.rots_zyx_to_legacy_aspire(rots)

    imgs_clean = vols.project(rots).asnumpy()
    assert np.allclose(results, imgs_clean, atol=1e-7)


def test_rotate_axes(res, dtype):
    # In this test we instantiate Volume instance `vol`, containing a single nonzero
    # voxel in the first octant, and rotate it by multiples of pi/2 about each axis.
    # We then compare to reference volumes containing appropriately located nonzero voxel.

    # Create a Volume instance to rotate.
    # This volume has a value of 1 in the first octant at (1, 1, 1) and zeros elsewhere.
    L = res
    data = np.zeros((L, L, L), dtype=dtype)
    data[L // 2 + 1, L // 2 + 1, L // 2 + 1] = 1
    vol = Volume(data)

    # Create a dict with map from axis and angle of rotation to new location (z, y, x) of nonzero voxel.
    ref_pts = {
        ("x", 0): (1, 1, 1),
        ("x", pi / 2): (1, -1, 1),
        ("x", pi): (-1, -1, 1),
        ("x", 3 * pi / 2): (-1, 1, 1),
        ("y", 0): (1, 1, 1),
        ("y", pi / 2): (-1, 1, 1),
        ("y", pi): (-1, 1, -1),
        ("y", 3 * pi / 2): (1, 1, -1),
        ("z", 0): (1, 1, 1),
        ("z", pi / 2): (1, 1, -1),
        ("z", pi): (1, -1, -1),
        ("z", 3 * pi / 2): (1, -1, 1),
    }

    center = np.array([L // 2] * 3)

    # Rotate Volume 'vol' and test against reference volumes.
    axes = ["x", "y", "z"]
    angles = [0, pi / 2, pi, 3 * pi / 2]
    for axis, angle in product(axes, angles):
        # Build rotation matrices
        rot_mat = Rotation.about_axis(axis, angle, dtype=dtype)

        # Rotate Volume 'vol' by rotations 'rot_mat'
        rot_vol = vol.rotate(rot_mat, zero_nyquist=False)

        # Build reference volumes using dict 'ref_pts'
        ref_vol = np.zeros((L, L, L), dtype=dtype)
        # Assign the location of non zero voxel
        loc = center + np.array(ref_pts[axis, angle])
        ref_vol[tuple(loc)] = 1

        # Test that rotated volumes align with reference volumes
        assert np.allclose(ref_vol, rot_vol, atol=utest_tolerance(dtype))


def test_rotate(vols_hot_cold):
    """
    We rotate Volumes containing random hot/cold spots by random rotations and check that
    hot/cold spots in the rotated Volumes are in the expected locations.
    """
    vols, hot_cold_locs, vol_center = vols_hot_cold
    dtype = vols.dtype
    L = vols.resolution

    # Generate random rotations.
    rots = Rotation.generate_random_rotations(n=vols.n_vols, dtype=dtype)

    # Expected location of hot/cold spots relative to (0, 0, 0) origin in (x, y, z) order.
    # Note, we write the simpler `(x, y, z) @ rots.T` in place of `(rots @ (x, y, z).T).T`
    expected_hot_cold = hot_cold_locs[..., ::-1] @ rots.invert().matrices

    # Expected location of hot/cold spots relative to Volume center (L/2, L/2, L/2) in (z, y, x) order.
    expected_locs = np.round(expected_hot_cold[..., ::-1] + vol_center)

    # Rotate Volumes.
    rotated_vols = vols.rotate(rots)

    # Check that new hot/cold spots are within 1 pixel of expectecd locations.
    for i in range(vols.n_vols):
        v = rotated_vols.asnumpy()[i]
        new_hot_loc = np.unravel_index(np.argmax(v), (L, L, L))
        new_cold_loc = np.unravel_index(np.argmin(v), (L, L, L))
        np.testing.assert_allclose(new_hot_loc, expected_locs[i, 0], atol=1)
        np.testing.assert_allclose(new_cold_loc, expected_locs[i, 1], atol=1)


def test_rotate_broadcast_unicast(asym_vols):
    # Build `Rotation` objects. A singleton for broadcasting and a stack for unicasting.
    # The stack consists of copies of the singleton.
    dtype = asym_vols.dtype
    rot = Rotation.generate_random_rotations(n=1, seed=1234, dtype=dtype)
    rots = Rotation(np.broadcast_to(rot.matrices, (asym_vols.n_vols, 3, 3)))

    # Broadcast the singleton `Rotation` across the `Volume` stack.
    vols_broadcast = asym_vols.rotate(rot)

    # Unicast the `Rotation` stack across the `Volume` stack.
    vols_unicast = asym_vols.rotate(rots)

    # Tests that all volumes match.
    assert np.allclose(vols_broadcast, vols_unicast, atol=utest_tolerance(dtype))


def to_vec(vols_1, vec):
    """Compute the to_vec method and compare."""
    result = vols_1.to_vec()
    assert result == vec
    assert isinstance(result, np.ndarray)


def test_from_vec(vec, vols_1):
    """Compute Volume from_vec method and compare."""
    vol = Volume.from_vec(vec)
    assert np.allclose(vol, vols_1)
    assert isinstance(vol, Volume)


def test_vec_id1(vols_1):
    """Test composition of from_vec(to_vec)."""
    # Construct vec
    vec = vols_1.to_vec()

    # Convert back to Volume and compare
    assert np.allclose(Volume.from_vec(vec), vols_1)


def test_vec_id2(vec):
    """Test composition of to_vec(from_vec)."""
    # Construct Volume
    vol = Volume.from_vec(vec)

    # # Convert back to vec and compare
    assert np.all(vol.to_vec() == vec)


def test_transpose(data_1, vols_1):
    data_t = np.transpose(data_1, (0, 3, 2, 1))

    result = vols_1.transpose()
    assert np.all(result == data_t)
    assert isinstance(result, Volume)

    result = vols_1.T
    assert np.all(result == data_t)
    assert isinstance(result, Volume)


def test_flatten(vols_1, data_1):
    result = vols_1.flatten()
    assert np.all(result == data_1.flatten())
    assert isinstance(result, np.ndarray)


def test_flip(vols_1, data_1):
    # Test over all sane axis.
    for axis in powerset(range(1, 4)):
        if not axis:
            # test default
            result = vols_1.flip()
            axis = 1
        else:
            result = vols_1.flip(axis)
        assert np.all(result == np.flip(data_1, axis))
        assert isinstance(result, Volume)

    # Test axis 0 raises
    msg = r"Cannot flip axis 0: stack axis."
    with raises(ValueError, match=msg):
        _ = vols_1.flip(axis=0)

    with raises(ValueError, match=msg):
        _ = vols_1.flip(axis=(0, 1))


def test_downsample(res):
    vols = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")), pixel_size=1.23
    )
    result = vols.downsample(res)
    og_res = vols.resolution
    ds_res = result.resolution

    # Confirm the pixel size is scaled
    np.testing.assert_approx_equal(
        result.pixel_size,
        vols.pixel_size * og_res / ds_res,
        err_msg="Incorrect pixel size.",
    )

    # check signal energy
    np.testing.assert_allclose(
        anorm(vols.asnumpy(), axes=(1, 2, 3)) / og_res,
        anorm(result.asnumpy(), axes=(1, 2, 3)) / ds_res,
        atol=1e-3,
    )

    # check gridpoints
    np.testing.assert_allclose(
        vols.asnumpy()[:, og_res // 2, og_res // 2, og_res // 2],
        result.asnumpy()[:, ds_res // 2, ds_res // 2, ds_res // 2],
        atol=1e-4,
    )


def test_shape(vols_1, res):
    assert vols_1.shape == (N, res, res, res)
    assert vols_1.stack_shape == (N,)
    assert vols_1.stack_ndim == 1
    assert vols_1.n_vols == N


def test_multi_dim_shape(vols_12, res):
    assert vols_12.shape == (2, N, res, res, res)
    assert vols_12.stack_shape == (2, N)
    assert vols_12.stack_ndim == 2
    assert vols_12.n_vols == 2 * N


def test_bad_key(vols_12):
    with raises(ValueError, match=r"slice length must be"):
        _ = vols_12[tuple(range(vols_12.ndim + 1))]


def test_multi_dim_gets(vols_12, data_1, data_2):
    assert np.allclose(vols_12[0], data_1)
    # Test a slice
    assert np.allclose(vols_12[1, 1:], data_2[1:])


def test_multi_dim_sets(vols_12, data_1, data_2):
    vols_12[0, 1] = 123
    # Check the values changed
    assert np.allclose(vols_12[0, 1], 123)
    # and only those values changed
    assert np.allclose(vols_12[0, 0], data_1[0])
    assert np.allclose(vols_12[0, 2:], data_1[2:])
    assert np.allclose(vols_12[1, :], data_2)


def test_multi_dim_sets_slice(vols_12, data_1, data_2):
    vols_12[0, 1:] = 456
    # Check the values changed
    assert np.allclose(vols_12[0, 1:], 456)
    # and only those values changed
    assert np.allclose(vols_12[0, 0], data_1[0])
    assert np.allclose(vols_12[1, :], data_2)


def test_multi_dim_reshape(vols_12, data_12, res):
    X = vols_12.stack_reshape(N, 2)
    # Compare with np.reshape of stack axes of ndarray
    assert np.allclose(X, data_12.reshape(N, 2, res, res, res))
    # and as tuples
    Y = vols_12.stack_reshape((N, 2))
    assert np.allclose(X, Y)


def test_multi_dim_flattens(vols_12, data_12, res):
    X = vols_12.stack_reshape(2 * N)
    assert np.allclose(X, data_12.reshape(-1, res, res, res))
    # and as tuples
    Y = vols_12.stack_reshape((2 * N,))
    np.allclose(X, Y)


def test_multi_dim_flattens_trick(vols_12, data_12, res):
    X = vols_12.stack_reshape(-1)
    assert np.allclose(X, data_12.reshape(-1, res, res, res))
    # and as tuples
    Y = vols_12.stack_reshape((-1,))
    assert np.allclose(X, Y)


def test_multi_dim_bad_reshape(vols_12):
    # Incorrect flat shape
    with raises(ValueError, match=r"Number of volumes"):
        _ = vols_12.stack_reshape(8675309)

    # Incorrect mdin shape
    with raises(ValueError, match=r"Number of volumes"):
        _ = vols_12.stack_reshape(42, 8675309)


def test_multi_dim_broadcast(data_12, data_1, data_2):
    X = data_12 + data_1
    assert np.allclose(X[0], 2 * data_1)
    assert np.allclose(X[1], data_1 + data_2)


def test_asnumpy_readonly():
    """
    Attempting assignment should raise an error.
    """
    ary = np.random.random((3, 8, 8, 8))
    im = Volume(ary)
    vw = im.asnumpy()

    # Attempt assignment
    with raises(ValueError, match=r".*destination is read-only.*"):
        vw[0, 0, 0, 0] = 123


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_project_broadcast(dtype):
    L = 32

    # Create stack of 3 Volumes.
    n_vols = 3
    vols = AsymmetricVolume(L=L, C=n_vols, dtype=dtype).generate()

    # Create a singleton and stack of Rotations.
    rot = Rotation.about_axis("z", np.pi / 6, dtype=dtype)
    rots = Rotation.about_axis("z", [np.pi / 4, np.pi / 3, np.pi / 2], dtype=dtype)

    # Test mask.
    mask = grid_2d(L)["r"] < 1

    # Broadcast Volume stack with singleton Rotation and compare with individually generated projections.
    projs_3_1 = vols.project(rot).asnumpy()
    for i in range(n_vols):
        proj_i = vols[i].project(rot).asnumpy()
        assert np.allclose(
            projs_3_1[i][mask],
            proj_i[0][mask],
            atol=utest_tolerance(dtype),
        )

    assert projs_3_1.shape[0] == n_vols

    # Broadcast Volume stack with Rotation stack of same size and compare with individually generated projections.
    projs_3_3 = vols.project(rots).asnumpy()
    for i in range(n_vols):
        proj_i = vols[i].project(rots[i]).asnumpy()
        assert np.allclose(
            projs_3_3[i][mask],
            proj_i[0][mask],
            atol=utest_tolerance(dtype),
        )

    assert projs_3_3.shape[0] == n_vols

    # Note: The test case for a single Volume and a stack of Rotations is covered above in testProject.

    # Check we raise an error for incompatible stack sizes.
    msg = "Cannot broadcast with 2 Rotations and 3 Volumes."
    with raises(NotImplementedError, match=msg):
        _ = vols.project(rots[:2])


# SYM_GROUP_PARAMS consists of (initializing method, string representation).
# Testing just the basic cases of setting the symmetry group from
# a SymmetryGroup instance, a string, and the default.
SYM_GROUP_PARAMS = [(TSymmetryGroup(np.float32), "T"), ("D2", "D2"), (None, "C1")]


@pytest.mark.parametrize("sym_group, sym_string", SYM_GROUP_PARAMS)
def test_symmetry_group_set_get(sym_group, sym_string):
    L = 8
    dtype = np.float32
    data = np.arange(L**3, dtype=dtype).reshape(L, L, L)
    vol = Volume(data, symmetry_group=sym_group, dtype=dtype)

    # Check Volume symmetry_group.
    assert isinstance(vol.symmetry_group, SymmetryGroup)
    assert str(vol.symmetry_group) == sym_string


def test_symmetry_group_pass_through(symmetric_vols):
    vol_c3, _ = symmetric_vols
    sym_group = str(vol_c3.symmetry_group)
    assert sym_group == "C3"

    # Check symmetry_group pass-through for various transformations.
    assert str(vol_c3.astype(np.float64).symmetry_group) == sym_group  # astype
    assert str(vol_c3[0].symmetry_group) == sym_group  # getitem
    assert (
        str(vol_c3.stack_reshape((1, 1)).symmetry_group) == sym_group
    )  # stack_reshape
    assert (
        str(vol_c3.downsample(vol_c3.resolution // 2).symmetry_group) == sym_group
    )  # downsample


def test_transformation_symmetry_warnings(symmetric_vols):
    """
    A warning should be emitted for transpose, flip, and rotate.
    """
    vol_c3, _ = symmetric_vols
    sym_group = str(vol_c3.symmetry_group)
    assert sym_group == "C3"

    # Check we get warning for each transformation.
    with pytest.warns(
        UserWarning, match=r".*`symmetry_group` attribute is being set to `C1`.*"
    ) as record:
        vol_t = vol_c3.T
        vol_f = vol_c3.flip()
        vol_r = vol_c3.rotate(Rotation.about_axis("x", np.pi, dtype=vol_c3.dtype))
    assert len(record) == 3

    # Check symmetry_group has been set to C1.
    assert str(vol_t.symmetry_group) == "C1"
    assert str(vol_f.symmetry_group) == "C1"
    assert str(vol_r.symmetry_group) == "C1"

    # Check original volume has retained C3 symmetry.
    assert str(vol_c3.symmetry_group) == "C3"


def test_aglebraic_ops_symmetry_warnings(symmetric_vols):
    """
    A warning should be emitted for  add, sub, mult, and div.
    """
    vol_c3, vol_c4 = symmetric_vols

    # Compatible symmetry should retain symmetry_group and emit no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (vol_c3 + vol_c3).symmetry_group == vol_c3.symmetry_group
        assert (vol_c3 - vol_c3).symmetry_group == vol_c3.symmetry_group
        assert (vol_c3 * vol_c3).symmetry_group == vol_c3.symmetry_group
        assert (
            vol_c3 / (vol_c3 + 1)
        ).symmetry_group == vol_c3.symmetry_group  # plus 1 to avoid division by 0.

    # Incompatible symmetry should warn and set symmetry_group to C1.
    with pytest.warns(
        UserWarning, match=r".*`symmetry_group` attribute is being set to `C1`.*"
    ) as record:
        vols_sum = vol_c3 + vol_c4
        vol_array_diff = vol_c3 - vol_c4.asnumpy()
        vols_mult = vol_c3 * vol_c4
        vol_array_div = vol_c3 / (vol_c4.asnumpy() + 1)

    assert str(vols_sum.symmetry_group) == "C1"
    assert str(vol_array_diff.symmetry_group) == "C1"
    assert str(vols_mult.symmetry_group) == "C1"
    assert str(vol_array_div.symmetry_group) == "C1"

    # Should have 4 warnings on record.
    assert len(record) == 4


def test_volume_load_with_symmetry():
    # Check we can load a Volume with symmetry_group.
    vol = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_down8.npy")),
        symmetry_group="C3",
    )
    assert isinstance(vol.symmetry_group, CnSymmetryGroup)
    assert str(vol.symmetry_group) == "C3"
