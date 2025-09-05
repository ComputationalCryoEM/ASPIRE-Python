import numpy as np
import pytest

from aspire.abinitio import (
    CLSymmetryC2,
    CLSymmetryC3C4,
    CLSymmetryCn,
    _cl_angles_to_ind,
    build_outer_products,
    g_sync,
)
from aspire.abinitio.commonline_cn import MeanOuterProductEstimator
from aspire.source import Simulation
from aspire.utils import (
    J_conjugate,
    Rotation,
    all_pairs,
    cyclic_rotations,
    mean_aligned_angular_distance,
)
from aspire.volume import CnSymmetricVolume

# A set of these parameters are marked expensive to reduce testing time.
# Each tuple holds the parameters (n_img, resolution "L", cyclic order "order", dtype).
param_list_c2 = [(55, 44, 2, np.float32)]

param_list_c3_c4 = [
    (24, 44, 3, np.float32),
    (24, 45, 4, np.float64),
    pytest.param(24, 44, 4, np.float32, marks=pytest.mark.expensive),
    pytest.param(24, 44, 3, np.float64, marks=pytest.mark.expensive),
    pytest.param(24, 44, 4, np.float64, marks=pytest.mark.expensive),
    pytest.param(24, 45, 3, np.float32, marks=pytest.mark.expensive),
    pytest.param(24, 45, 4, np.float32, marks=pytest.mark.expensive),
    pytest.param(24, 45, 3, np.float64, marks=pytest.mark.expensive),
]

# For testing Cn methods where n>4.
param_list_cn = [
    (8, 44, 5, np.float32),
    pytest.param(24, 45, 6, np.float64, marks=pytest.mark.expensive),
    pytest.param(24, 44, 7, np.float32, marks=pytest.mark.expensive),
    pytest.param(24, 44, 8, np.float32, marks=pytest.mark.expensive),
    pytest.param(24, 45, 9, np.float64, marks=pytest.mark.expensive),
]


# Method to instantiate a Simulation source and orientation estimation object.
def source_orientation_objs(n_img, L, order, dtype):
    # This Volume is hand picked to have a fairly even distribution of density.
    # Due to the rotations used to generate symmetric volumes, some seeds will
    # generate volumes with a high concentration of denisty in the center causing
    # misidentification of common-lines.
    vol = CnSymmetricVolume(
        L=L,
        C=1,
        K=50,
        order=order,
        seed=65,
        dtype=dtype,
    ).generate()

    angles = None
    if order > 4:
        # We artificially exclude equator images from the simulation as they will be
        # incorrectly identified by the CL method. We keep images slightly further away
        # from being equator images than the 10 degree default threshold used in the CL method.
        rotations, _ = CLSymmetryCn.generate_candidate_rots(
            n=n_img,
            equator_threshold=15,
            order=order,
            degree_res=1,
            seed=123,  # Generate different rotations than candidates used in CL method.
        )
        angles = Rotation(rotations).angles

    seed = 1
    src = Simulation(
        L=L,
        n=n_img,
        offsets=0,
        amplitudes=1,
        dtype=dtype,
        vols=vol,
        angles=angles,
        C=1,
        seed=seed,
    )

    # Use default n_theta = 360.
    cl_kwargs = dict(
        src=src,
        max_shift=1 / L,
        seed=seed,
        mask=False,
    )

    if order in [3, 4]:
        cl_class = CLSymmetryC3C4
        cl_kwargs["symmetry"] = f"C{order}"
    elif order == 2:
        cl_class = CLSymmetryC2
        cl_kwargs["min_dist_cls"] = 15
    else:
        cl_class = CLSymmetryCn
        cl_kwargs["symmetry"] = f"C{order}"
    orient_est = cl_class(**cl_kwargs)

    return src, orient_est


@pytest.mark.parametrize(
    "n_img, L, order, dtype", param_list_c2 + param_list_c3_c4 + param_list_cn
)
def test_estimate_rotations(n_img, L, order, dtype):
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)

    # Estimate rotations.
    cl_symm.estimate_rotations()
    rots_est = cl_symm.rotations

    # Ground truth rotations.
    rots_gt = src.rotations

    # g-synchronize ground truth rotations.
    rots_gt_sync = g_sync(rots_est, order, rots_gt)

    # Register estimates to ground truth rotations and check that the
    # mean angular distance between them is less than 3 degrees.
    mean_aligned_angular_distance(rots_est, rots_gt_sync, degree_tol=3)


@pytest.mark.parametrize("n_img, L, order, dtype", param_list_c3_c4)
def test_relative_rotations(n_img, L, order, dtype):
    # Simulation source and common lines estimation instance
    # corresponding to volume with C3 or C4 symmetry.
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)

    # Estimate relative viewing directions.
    cl_symm.build_clmatrix()
    cl = cl_symm.clmatrix
    Rijs = cl_symm._estimate_all_Rijs_c3_c4(cl)

    # Each Rij belongs to the set {Ri.Tg_n^sRj, JRi.Tg_n^sRjJ},
    # s = 1, 2, ..., order. We find the mean squared error over
    # the minimum error between Rij and the above set.
    gs = cyclic_rotations(order, dtype).matrices
    rots_gt = src.rotations

    # Find the angular distance between each Rij and the ground truth.
    pairs = all_pairs(n_img)
    angular_distance = np.zeros(len(pairs))
    for idx, (i, j) in enumerate(pairs):
        Rij = Rijs[idx]
        Rij_J = J_conjugate(Rij)
        Ri_gt = rots_gt[i]
        Rj_gt = rots_gt[j]
        dist = np.zeros(order)
        for s in range(order):
            Rij_s_gt = Ri_gt.T @ gs[s] @ Rj_gt
            dist[s] = np.minimum(
                Rotation.angle_dist(Rij, Rij_s_gt),
                Rotation.angle_dist(Rij_J, Rij_s_gt),
            )
        angular_distance[idx] = np.min(dist)
    mean_angular_distance = np.mean(angular_distance) * 180 / np.pi

    # Assert that the mean_angular_distance is less than 5 degrees.
    assert mean_angular_distance < 5


@pytest.mark.parametrize("n_img, L, order, dtype", param_list_c3_c4)
def test_self_relative_rotations(n_img, L, order, dtype):
    # Simulation source and common lines Class corresponding to
    # volume with C3 or C4 symmetry.
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)

    # Estimate self-relative viewing directions, Riis.
    scl = cl_symm._self_clmatrix_c3_c4()
    Riis = cl_symm._estimate_all_Riis_c3_c4(scl)

    # Each estimated Rii belongs to the set
    # {Ri.Tg_nRi, Ri.Tg_n^{n-1}Ri, JRi.Tg_nRiJ, JRi.Tg_n^{n-1}RiJ}.
    # We find the minimum angular distance between the estimate Rii
    # and the 4 possible ground truths.
    rots_symm = cyclic_rotations(order, dtype).matrices
    g = rots_symm[1]
    rots_gt = src.rotations

    # Find angular distance between estimate and ground truth.
    dist = np.zeros(4)
    angular_distance = np.zeros(n_img)
    for i, rot_gt in enumerate(rots_gt):
        Rii_gt = rot_gt.T @ g @ rot_gt
        Rii = Riis[i]
        cases = np.array([Rii, Rii.T, J_conjugate(Rii), J_conjugate(Rii.T)])
        for i, estimate in enumerate(cases):
            dist[i] = Rotation.angle_dist(estimate, Rii_gt)
        angular_distance[i] = min(dist)
    mean_angular_distance = np.mean(angular_distance) * 180 / np.pi

    # Check that mean_angular_distance is less than 5 degrees.
    assert mean_angular_distance < 5


@pytest.mark.parametrize("n_img, L, order, dtype", param_list_c3_c4 + param_list_cn)
def test_relative_viewing_directions(n_img, L, order, dtype):
    # Simulation source and common lines Class corresponding to
    # volume with C3 or C4 symmetry.
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)

    # Calculate ground truth relative viewing directions, viis and vijs.
    rots_gt = src.rotations

    viis_gt = np.zeros((n_img, 3, 3))
    for i in range(n_img):
        vi = rots_gt[i, 2]
        viis_gt[i] = np.outer(vi, vi)

    pairs = all_pairs(n_img)
    n_pairs = len(pairs)
    vijs_gt = np.zeros((n_pairs, 3, 3))
    for idx, (i, j) in enumerate(pairs):
        vi = rots_gt[i, 2]
        vj = rots_gt[j, 2]
        vijs_gt[idx] = np.outer(vi, vj)

    # Estimate relative viewing directions.
    vijs, viis = cl_symm._estimate_relative_viewing_directions()

    # Since ground truth vijs and viis are rank 1 matrices they span a 1D subspace.
    # We use SVD to find this subspace for our estimates and the ground truth relative viewing directions.
    # We then calculate the angular distance between these subspaces (and take the mean).
    # SVD's:
    uij_gt, _, _ = np.linalg.svd(vijs_gt)
    uii_gt, _, _ = np.linalg.svd(viis_gt)
    uij_est, sij, _ = np.linalg.svd(vijs)
    uii_est, sii, _ = np.linalg.svd(viis)
    uij_J_est, _, _ = np.linalg.svd(J_conjugate(vijs))
    uii_J_est, _, _ = np.linalg.svd(J_conjugate(viis))

    # Ground truth 1D supbspaces.
    uij_gt = uij_gt[:, :, 0]
    uii_gt = uii_gt[:, :, 0]

    # 1D subspace of estimates.
    uij_est = uij_est[:, :, 0]
    uii_est = uii_est[:, :, 0]
    uij_J_est = uij_J_est[:, :, 0]
    uii_J_est = uii_J_est[:, :, 0]

    # Calculate angular distance between subspaces.
    theta_vij = np.arccos(np.sum(uij_gt * uij_est, axis=1))
    theta_vij_J = np.arccos(np.sum(uij_gt * uij_J_est, axis=1))
    theta_vii = np.arccos(np.sum(uii_gt * uii_est, axis=1))
    theta_vii_J = np.arccos(np.sum(uii_gt * uii_J_est, axis=1))

    # Minimum angle between subspaces.
    min_theta_vij = np.min(
        (theta_vij, theta_vij_J, np.pi - theta_vij, np.pi - theta_vij_J), axis=0
    )
    min_theta_vii = np.min(
        (theta_vii, theta_vii_J, np.pi - theta_vii, np.pi - theta_vii_J), axis=0
    )

    # Calculate the mean minimum angular distance.
    angular_dist_vijs = np.mean(min_theta_vij)
    angular_dist_viis = np.mean(min_theta_vii)

    # Check that estimates are indeed approximately rank-1.
    # ie. check that the svd is close to [1, 0, 0].
    sii = (
        sii / np.linalg.norm(sii, axis=1)[..., np.newaxis]
    )  # Normalize for comparison to [1, 0, 0]
    sij = (
        sij / np.linalg.norm(sij, axis=1)[..., np.newaxis]
    )  # Normalize for comparison to [1, 0, 0]
    error_ij = np.linalg.norm(np.array([1, 0, 0], dtype=dtype) - sij, axis=1)
    error_ii = np.linalg.norm(np.array([1, 0, 0], dtype=dtype) - sii, axis=1)
    max_tol_ij = 1e-7
    mean_tol_ij = 1e-7
    # For order < 5, the method for estimating vijs leads to estimates
    # which do not as tightly approximate rank-1.
    if order < 5:
        max_tol_ij = 4e-1
        mean_tol_ij = 4e-3
    assert np.max(error_ij) < max_tol_ij
    assert np.max(error_ii) < 1e-6
    assert np.mean(error_ij) < mean_tol_ij
    assert np.mean(error_ii) < 1e-7

    # Check that the mean angular difference is within 2 degrees.
    angle_tol = 2 * np.pi / 180
    if order > 4:
        angle_tol = 4 * np.pi / 180

    assert angular_dist_vijs < angle_tol
    assert angular_dist_viis < angle_tol


@pytest.mark.parametrize("n_img, L, order, dtype", param_list_c3_c4)
def test_self_commonlines(n_img, L, order, dtype):
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)
    n_theta = cl_symm.n_theta

    # Initialize common-lines orientation estimation object and compute self-common-lines matrix.
    scl = cl_symm._self_clmatrix_c3_c4()

    # Compute ground truth self-common-lines matrix.
    rots = src.rotations
    scl_gt = build_self_commonlines_matrix(n_theta, rots, order)

    # Since we search for self common lines whose angle differences fall
    # outside of 180 degrees by a tolerance of 2 * (360 // L), we must exclude
    # indices whose ground truth self common lines fall within that tolerance.
    gt_diffs = abs(scl_gt[:, 0] - scl_gt[:, 1])
    res = 2 * (360 // L)
    good_indices = (gt_diffs < (180 - res)) | (gt_diffs > (180 + res))
    scl = scl[good_indices]
    scl_gt = scl_gt[good_indices]

    # Get angle difference between scl_gt and scl.
    scl_diff1 = scl_gt - scl
    scl_diff2 = scl_gt - np.flip(scl, 1)  # Order of indices might be switched.
    scl_diff1_angle = scl_diff1 * 2 * np.pi / n_theta
    scl_diff2_angle = scl_diff2 * 2 * np.pi / n_theta

    # cosine is invariant to 2pi, and abs is invariant to +-pi due to J-conjugation.
    # We take the mean deviation wrt to the two lines in each image.
    scl_diff1_angle_mean = np.mean(np.arccos(abs(np.cos(scl_diff1_angle))), axis=1)
    scl_diff2_angle_mean = np.mean(np.arccos(abs(np.cos(scl_diff2_angle))), axis=1)

    scl_diff_angle_mean = np.vstack((scl_diff1_angle_mean, scl_diff2_angle_mean))
    scl_idx = np.argmin(scl_diff_angle_mean, axis=0)
    min_mean_angle_diff = scl_idx.choose(scl_diff_angle_mean)

    # Assert scl detection rate is 100% for 5 degree angle tolerance
    angle_tol_err = 5 * np.pi / 180
    detection_rate = np.count_nonzero(min_mean_angle_diff < angle_tol_err) / len(scl)
    assert np.allclose(detection_rate, 1.0)


@pytest.mark.parametrize("n_img, L, order, dtype", param_list_c2)
def test_commonlines_c2(n_img, L, order, dtype):
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)
    n_theta = cl_symm.n_theta

    # Build common-lines matrix.
    cl_symm.build_clmatrix()
    cl = cl_symm.clmatrix

    # Ground truth common-lines matrix.
    cl_gt = _gt_cl_c2(n_theta, src.rotations)

    # Convert from indices to angles. Use angle of common-line in [0, 180).
    cl = (cl * 360 / n_theta) % 180
    cl_gt = (cl_gt * 360 / n_theta) % 180

    pairs = all_pairs(n_img)
    angle_tol = 2  # degrees
    within_tol = 0
    for i, j in pairs:
        # For each pair of images the two sets of mutual common-lines in cl, (cl[0,i,j], cl[0,j,i])
        # and (cl[1,i,j], cl[1,j,i]), should each match one of the two sets in the ground truth cl_gt.
        # We take the sum of errors from both combinations.
        err_1 = (
            abs(cl[0, i, j] - cl_gt[0, i, j])
            + abs(cl[0, j, i] - cl_gt[0, j, i])
            + abs(cl[1, i, j] - cl_gt[1, i, j])
            + abs(cl[1, j, i] - cl_gt[1, j, i])
        )
        err_2 = (
            abs(cl[0, i, j] - cl_gt[1, i, j])
            + abs(cl[0, j, i] - cl_gt[1, j, i])
            + abs(cl[1, i, j] - cl_gt[0, i, j])
            + abs(cl[1, j, i] - cl_gt[0, j, i])
        )
        min_err = min(err_1, err_2)
        if min_err <= angle_tol:
            within_tol += 1

    # Check that at least 90% of estimates are within `angle_tol` degrees.
    assert within_tol / len(pairs) > 0.90


@pytest.mark.parametrize("n_img, L, order, dtype", param_list_c3_c4)
def test_commonlines(n_img, L, order, dtype):
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)
    n_theta = cl_symm.n_theta

    # Build common-lines matrix.
    cl_symm.build_clmatrix()
    cl = cl_symm.clmatrix

    # Compare common-line indices with ground truth angles.
    rots = src.rotations  # ground truth rotations
    rots_symm = cyclic_rotations(order, dtype).matrices
    pairs = all_pairs(n_img)
    within_1_degree = 0
    within_5_degrees = 0
    for i, j in pairs:
        a_ij_s = np.zeros(order)
        a_ji_s = np.zeros(order)
        # Convert common-line indices to angles. Use angle of common line in [0, 180).
        cl_ij = (cl[i, j] * 360 / n_theta) % 180
        cl_ji = (cl[j, i] * 360 / n_theta) % 180

        # The common-line estimates cl_ij, cl_ji should match the
        # true common-line angles a_ij_s, a_ji_s for some value s,
        # where s is the number of common-lines induced by the symmetric self.order.
        for s in range(order):
            rel_rot = rots[i].T @ rots_symm[s] @ rots[j]
            a_ij_s[s] = np.rad2deg(np.arctan(-rel_rot[0, 2] / rel_rot[1, 2])) % 180
            a_ji_s[s] = np.rad2deg(np.arctan(-rel_rot[2, 0] / rel_rot[2, 1])) % 180
        best_s = np.argmin(abs(cl_ij - a_ij_s) + abs(cl_ji - a_ji_s))
        diff_ij = abs(cl_ij - a_ij_s[best_s])
        diff_ji = abs(cl_ji - a_ji_s[best_s])

        # Count the number of good estimates.
        if diff_ij < 1:
            within_1_degree += 1
            within_5_degrees += 1
        elif diff_ij < 5:
            within_5_degrees += 1

        if diff_ji < 1:
            within_1_degree += 1
            within_5_degrees += 1
        elif diff_ji < 5:
            within_5_degrees += 1

    # Assert that at least 98% of estimates are within 5 degrees and
    # at least 90% of estimates are within 1 degree.
    n_estimates = 2 * len(pairs)
    within_5 = within_5_degrees / n_estimates
    within_1 = within_1_degree / n_estimates
    assert within_5 > 0.98
    assert within_1 > 0.90


@pytest.mark.parametrize(
    "n_img, dtype",
    [(3, np.float32), (3, np.float64), (20, np.float32), (20, np.float64)],
)
def test_global_J_sync(n_img, dtype):
    """
    For this test we build a set of 3rd row outer products, vijs and viis, and
    randomly J_conjugate them. We then test that J-synchronization is correct up
    to conjugation of the entire set. To expose bugs in the implementation, we use
    n_img = 3 as the smallest possible example to run the algorithm, which computes
    relative handedness over all triplets of images (in this case 1 triplet).
    """
    L = 16  # test not dependent on L
    order = 3  # test not dependent on order
    _, orient_est = source_orientation_objs(n_img, L, order, dtype)

    # Build a set of outer products of random third rows.
    vijs, viis, _ = build_outer_products(n_img, dtype)

    # J-conjugate some of these outer products (every other element).
    vijs_conj, viis_conj = vijs.copy(), viis.copy()
    inds_ij = np.random.choice(len(vijs), size=len(vijs) // 2, replace=False)
    inds_ii = np.random.choice(len(viis), size=len(viis) // 2, replace=False)
    vijs_conj[inds_ij] = J_conjugate(vijs[inds_ij])
    viis_conj[inds_ii] = J_conjugate(viis[inds_ii])

    # Synchronize vijs_conj and viis_conj.
    # Note: `_global_J_sync()` does not depend on cyclic order, so we can use
    # either cl_orient_ests[3] or cl_orient_ests[4] to access the method.
    vijs_sync, viis_sync = orient_est._global_J_sync(vijs_conj, viis_conj)

    # Check that synchronized outer products equal original
    # up to J-conjugation of the entire set.
    need_to_conj_vijs = not np.allclose(vijs_sync[inds_ij][0], vijs[inds_ij][0])
    if need_to_conj_vijs:
        assert np.allclose(vijs, J_conjugate(vijs_sync))
        assert np.allclose(viis, J_conjugate(viis_sync))
    else:
        assert np.allclose(vijs, vijs_sync)
        assert np.allclose(viis, viis_sync)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_pass_through(dtype):
    L = 16
    n_img = 20
    order = 3  # test does not depend on order
    src, cl_symm = source_orientation_objs(n_img, L, order, dtype)
    assert src.dtype == cl_symm.dtype


def build_self_commonlines_matrix(n_theta, rots, order):
    # Construct rotatation matrices associated with cyclic order.
    rots_symm = cyclic_rotations(order, rots.dtype).matrices

    # Build ground truth self-common-lines matrix.
    scl_gt = np.zeros((len(rots), 2), dtype=rots.dtype)
    g = rots_symm[1]
    g_n = rots_symm[-1]
    for i, rot in enumerate(rots):
        Ri = rot

        U1 = Ri.T @ g @ Ri
        U2 = Ri.T @ g_n @ Ri

        c1 = np.array([-U1[1, 2], U1[0, 2]])
        c2 = np.array([-U2[1, 2], U2[0, 2]])

        theta_g = np.arctan2(c1[1], c1[0]) % (2 * np.pi)
        theta_gn = np.arctan2(c2[1], c2[0]) % (2 * np.pi)

        scl_gt[i, 0] = np.round(theta_g * n_theta / (2 * np.pi)) % n_theta
        scl_gt[i, 1] = np.round(theta_gn * n_theta / (2 * np.pi)) % n_theta

    return scl_gt


def test_mean_outer_product_estimator():
    """
    Manully run MeanOuterProductEstimator for prebaked inputs.
    """

    est = MeanOuterProductEstimator()

    # Test arrays with opposite conjugation.
    V = np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    V_J = np.array([[1, 1, -1], [3, 3, -3], [-5, -5, 5]])

    # Push two matrices with opposite conjugation.
    est.push(V)
    est.push(V_J)

    # synchronized_mean will J-conjugate the second entry prior to averaging.
    assert np.allclose(est.synchronized_mean(), V)

    # Push two more.
    est.push(V_J)
    est.push(V)

    # The resulting synchronized_mean should be V.
    assert np.allclose(est.synchronized_mean(), V)


def test_square_mask():
    n_shifts = 2
    x_len = 4
    y_len = 4
    data = np.ones((x_len, n_shifts, y_len), dtype=np.float32)

    # Test centered mask.
    ref = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=int)
    x, y = x_len // 2, y_len // 2
    mask = CLSymmetryC2._square_mask(data, x, y, dist=1)
    for shift in range(n_shifts):
        assert np.array_equal(mask[:, shift], ref)

    # Test mask near edge of box.
    ref = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=int)
    x, y = 0, 0
    mask = CLSymmetryC2._square_mask(data, x, y, dist=2)
    for shift in range(n_shifts):
        assert np.array_equal(mask[:, shift], ref)


def _gt_cl_c2(n_theta, rots_gt):
    n_imgs = len(rots_gt)
    gs = cyclic_rotations(2)
    clmatrix_gt = np.zeros((2, n_imgs, n_imgs))
    for i in range(n_imgs):
        Ri = rots_gt[i]
        for j in range(i + 1, n_imgs):
            Rj = rots_gt[j]
            for idx, g in enumerate(gs):
                U = Ri.T @ g @ Rj
                c1 = np.array([-U[1, 2], U[0, 2]])
                c2 = np.array([U[2, 1], -U[2, 0]])
                clmatrix_gt[idx, i, j] = _cl_angles_to_ind(c1[np.newaxis, :], n_theta)
                clmatrix_gt[idx, j, i] = _cl_angles_to_ind(c2[np.newaxis, :], n_theta)
    return clmatrix_gt
