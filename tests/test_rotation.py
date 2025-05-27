import logging
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sp_rot

from aspire.downloader import emdb_2660
from aspire.utils import Rotation, nearest_rotations, utest_tolerance

logger = logging.getLogger(__name__)


# Parameters

NUM_ROTS = 32
SEED = 0

DTYPES = [
    np.float32,
    np.float64,
]


# Fixtures


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def rot_obj(dtype):
    return Rotation.generate_random_rotations(NUM_ROTS, seed=SEED, dtype=dtype)


# Rotation Class Tests


def test_matrices(rot_obj):
    rot_ref = sp_rot.from_matrix(rot_obj.matrices)
    matrices = rot_ref.as_matrix()
    np.testing.assert_allclose(
        rot_obj.matrices, matrices, atol=utest_tolerance(rot_obj.dtype)
    )


def test_as_angles(rot_obj):
    rot_ref = sp_rot.from_euler("ZYZ", rot_obj.angles, degrees=False)
    angles = rot_ref.as_euler("ZYZ", degrees=False)
    np.testing.assert_allclose(rot_obj.angles, angles)


def test_from_matrix(rot_obj):
    rot_ref = sp_rot.from_matrix(rot_obj.matrices)
    angles = rot_ref.as_euler("ZYZ", degrees=False)
    rot = Rotation.from_matrix(rot_obj.matrices)
    np.testing.assert_allclose(rot.angles, angles)


def test_from_euler(rot_obj):
    rot_ref = sp_rot.from_euler("ZYZ", rot_obj.angles, degrees=False)
    matrices = rot_ref.as_matrix()
    rot = Rotation.from_euler(rot_obj.angles, dtype=rot_obj.dtype)
    np.testing.assert_allclose(rot._matrices, matrices)


def test_invert(rot_obj):
    rot_mat = rot_obj.matrices
    rot_mat_t = rot_obj.invert()
    np.testing.assert_allclose(rot_mat_t, np.transpose(rot_mat, (0, 2, 1)))


def test_multiplication(rot_obj):
    result = (rot_obj * rot_obj.invert()).matrices
    for i in range(len(rot_obj)):
        np.testing.assert_allclose(
            np.eye(3), result[i], atol=utest_tolerance(rot_obj.dtype)
        )


def test_register_rots(rot_obj):
    q_mat = Rotation.generate_random_rotations(1, dtype=rot_obj.dtype)[0]
    for flag in [0, 1]:
        regrots_ref = rot_obj.apply_registration(q_mat, flag)
        q_mat_est, flag_est = rot_obj.find_registration(regrots_ref)
        np.testing.assert_allclose(flag_est, flag)
        np.testing.assert_allclose(
            q_mat_est, q_mat, atol=utest_tolerance(rot_obj.dtype)
        )


def test_register(rot_obj):
    # These will yield two more distinct sets of random rotations wrt rot_obj
    set1 = Rotation.generate_random_rotations(NUM_ROTS, dtype=rot_obj.dtype)
    set2 = Rotation.generate_random_rotations(
        NUM_ROTS, dtype=rot_obj.dtype, seed=SEED + 7
    )
    # Align both sets of random rotations to rot_obj
    aligned_rots1 = rot_obj.register(set1)
    aligned_rots2 = rot_obj.register(set2)
    tol = utest_tolerance(rot_obj.dtype)
    np.testing.assert_array_less(aligned_rots1.mse(aligned_rots2), tol)
    np.testing.assert_array_less(aligned_rots2.mse(aligned_rots1), tol)


def test_mse(rot_obj):
    q_ang = [np.random.random(3)]
    q_mat = sp_rot.from_euler("ZYZ", q_ang, degrees=False).as_matrix()[0]
    for flag in [0, 1]:
        regrots_ref = rot_obj.apply_registration(q_mat, flag)
        mse = rot_obj.mse(regrots_ref)
        np.testing.assert_array_less(mse, utest_tolerance(rot_obj.dtype))


def test_common_lines(rot_obj):
    ell_ij, ell_ji = rot_obj.common_lines(8, 11, 360)
    np.testing.assert_equal([ell_ij, ell_ji], [235, 104])


def test_string(rot_obj):
    logger.debug(str(rot_obj))


def test_repr(rot_obj):
    logger.debug(repr(rot_obj))


def test_len(rot_obj):
    assert len(rot_obj) == NUM_ROTS


def test_setter_getter(rot_obj):
    # Excute set
    tmp = np.arange(9).reshape((3, 3))
    rot_obj[13] = tmp
    # Execute get
    np.testing.assert_equal(rot_obj[13], tmp)


def test_dtype(dtype, rot_obj):
    assert dtype == rot_obj.dtype


def test_about_axis_dtype(dtype):
    angles = np.random.uniform(0, 2 * np.pi, 10).astype(dtype, copy=False)
    rots = Rotation.about_axis("x", angles).matrices
    np.testing.assert_equal(rots.dtype, dtype)


def test_about_axis_float_dtype():
    angle = np.pi
    rot = Rotation.about_axis("y", angle).matrices
    np.testing.assert_equal(rot.dtype, np.float64)


def test_from_rotvec(rot_obj):
    # Build random rotation vectors.
    axis = np.array([1, 0, 0], dtype=rot_obj.dtype)
    angles = np.random.uniform(0, 2 * np.pi, 10)
    rot_vecs = np.array([angle * axis for angle in angles], dtype=rot_obj.dtype)

    # Build rotations using from_rotvec and about_axis (as reference).
    rotations = Rotation.from_rotvec(rot_vecs, dtype=rot_obj.dtype)
    ref_rots = Rotation.about_axis("x", angles, dtype=rot_obj.dtype)

    assert isinstance(rotations, Rotation)
    assert rotations.matrices.dtype == rot_obj.dtype
    np.testing.assert_allclose(rotations.matrices, ref_rots.matrices)


# Angular Distance Tests


def test_angle_dist(dtype):
    angles = np.array([i * np.pi / 360 for i in range(360)], dtype=dtype)
    rots = Rotation.about_axis("x", angles, dtype=dtype)

    # Calculate the angular distance between the identity, rots[0],
    # and rotations by multiples of pi/360 about the x-axis.
    # These should be equal to `angles`.
    angular_dist = Rotation.angle_dist(rots[0], rots, dtype)
    assert np.allclose(angles, angular_dist, atol=utest_tolerance(dtype))

    # Test incompatible shape error.
    with pytest.raises(ValueError, match=r"r1 and r2 are not broadcastable*"):
        _ = Rotation.angle_dist(rots[:3], rots[:5])

    # Test that single value returns as 0-dim.
    assert Rotation.angle_dist(rots[0], rots[1], dtype).ndim == 0


def test_mean_angular_distance(dtype):
    rots_z = Rotation.about_axis("z", [0, np.pi / 4, np.pi / 2], dtype=dtype).matrices
    rots_id = Rotation.about_axis("z", [0, 0, 0], dtype=dtype).matrices

    mean_ang_dist = Rotation.mean_angular_distance(rots_z, rots_id)

    assert np.allclose(mean_ang_dist, np.pi / 4)


def test_rot_with_refl(dtype):
    """
    Given random orthogonal matrices, test nearest_rotation returns a
    pure rotation which preserves our 2D projection operation.
    """

    N = 10  # Number of random rots
    SEED = 123
    L = 128  # Pixel size for projections
    atol = 1e-7  # Picked to cover both singles and doubles

    # Generate a sample of random rotations
    random_rot_mats = Rotation.generate_random_rotations(
        N, seed=SEED, dtype=dtype
    ).matrices

    # Sanity check we are starting with pure rotations
    np.testing.assert_allclose(np.linalg.det(random_rot_mats), 1, atol=atol)

    # Reflection operations
    refls = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [-1, -1, -1]], dtype=dtype)

    # Enumerate every combination of rot and refl into M
    M = np.empty((len(random_rot_mats) * len(refls), 3, 3), dtype=dtype)
    for i, (r, s) in enumerate(product(random_rot_mats, refls)):
        M[i] = r * s

    # Sanity check all entries are not pure rotations
    np.testing.assert_allclose(np.linalg.det(M), -1, atol=atol)

    R = nearest_rotations(M, allow_reflection=False)
    # Sanity check all entries are pure rotations
    np.testing.assert_allclose(np.linalg.det(R), 1, atol=atol)

    # Create a volume to use for projections
    v = emdb_2660().astype(dtype).downsample(L)

    # Project using set of transforms M
    ref_projections = v.project(M)

    # Project using set of rotations R
    rot_projections = v.project(R)

    # Validate we are equivalent
    np.testing.assert_allclose(rot_projections, ref_projections, atol=atol)
