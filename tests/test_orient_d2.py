import numpy as np
import pytest

from aspire.abinitio import CLSymmetryD2
from aspire.source import Simulation
from aspire.utils import (
    J_conjugate,
    Random,
    Rotation,
    all_pairs,
    mean_aligned_angular_distance,
    utest_tolerance,
)
from aspire.volume import DnSymmetricVolume, DnSymmetryGroup

##############
# Parameters #
##############

DTYPE = [np.float32, pytest.param(np.float64, marks=pytest.mark.expensive)]
RESOLUTION = [48, 49]
N_IMG = [10]
OFFSETS = [0, pytest.param(None, marks=pytest.mark.expensive)]

# Since these tests are optimized for runtime, detuned parameters cause
# the algorithm to be fickle, especially for small problem sizes.
# In particular, the parameters `grid_res`, inplane_res`, and `eq_min_dist`
# which control the number of candidate rotations used in the D2 algorithm
# will produce bad estimates if the candidates do not align closely with the
# ground truth rotations.
# This seed is chosen so the tests pass CI on github's envs as well
# as our self-hosted runner.
SEED = 3


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}", scope="module")
def resolution(request):
    return request.param


@pytest.fixture(params=N_IMG, ids=lambda x: f"n images={x}", scope="module")
def n_img(request):
    return request.param


@pytest.fixture(params=OFFSETS, ids=lambda x: f"offsets={x}", scope="module")
def offsets(request):
    return request.param


############
# Fixtures #
############


@pytest.fixture(scope="module")
def source(n_img, resolution, dtype, offsets):
    vol = DnSymmetricVolume(
        L=resolution, order=2, C=1, K=100, dtype=dtype, seed=SEED
    ).generate()

    src = Simulation(
        n=n_img,
        L=resolution,
        vols=vol,
        offsets=offsets,
        amplitudes=1,
        seed=SEED,
    )
    src = src.cache()  # Precompute image stack

    return src


@pytest.fixture(scope="module")
def orient_est(source):
    return build_cl_from_source(source)


#########
# Tests #
#########


def test_estimate_rotations(orient_est):
    """
    This test runs through the complete D2 algorithm and compares the
    estimated rotations to the ground truth rotations. In particular,
    we check that the estimates are close to the ground truth up to
    a local rotation by a D2 symmetry group member, a global J-conjugation,
    and a globally aligning rotation.
    """
    # Estimate rotations.
    orient_est.estimate_rotations()
    rots_est = orient_est.rotations

    # Ground truth rotations.
    rots_gt = orient_est.src.rotations

    # g-sync ground truth rotations.
    rots_gt_sync = g_sync_d2(rots_est, rots_gt)

    # Register estimates to ground truth rotations and check that the mean angular
    # distance between them is less than 5 degrees.
    mean_aligned_angular_distance(rots_est, rots_gt_sync, degree_tol=5)

    # Check dtype pass-through.
    assert rots_est.dtype == orient_est.dtype


def test_scl_scores(orient_est):
    """
    This test uses a Simulation generated with rotations taken directly
    from the D2 algorithm `sphere_grid` of candidate rotations. It is
    these candidates which should produce maximum correlation scores since
    they match perfectly the Simulation rotations.
    """
    # Generate lookup data and extract rotations from the candidate `sphere_grid`.
    # In this case, we take first 10 candidates from a non-equator viewing direction.
    orient_est._generate_lookup_data()
    cand_rots = orient_est.inplane_rotated_grid1
    non_eq_idx = int(np.argwhere(orient_est.eq_class1 == 0)[0][0])
    rots = cand_rots[non_eq_idx, :10]
    angles = Rotation(rots).angles

    # Create a Simulation using those first 10 candidate rotations.
    src = Simulation(
        n=orient_est.src.n,
        L=orient_est.src.L,
        vols=orient_est.src.vols,
        angles=angles,
        offsets=orient_est.src.offsets,
        amplitudes=1,
        seed=SEED,
    )

    # Initialize CL instance with new source.
    cl = build_cl_from_source(src)

    # Generate lookup data.
    cl._compute_shifted_pf()
    cl._generate_lookup_data()
    cl._generate_scl_lookup_data()

    # Compute self-commonline scores.
    cl._compute_scl_scores()

    # cl.scls_scores is shape (n_img, n_cand_rots). Since we used the first
    # 10 candidate rotations of the first non-equator viewing direction as our
    # Simulation rotations, the maximum correlation for image i should occur at
    # candidate rotation index (non_eq_idx * cl.n_inplane_rots + i).
    max_corr_idx = np.argmax(cl.scls_scores, axis=1)
    gt_idx = non_eq_idx * cl.n_inplane_rots + np.arange(10)

    # Check that self-commonline indices match ground truth.
    n_match = np.sum(max_corr_idx == gt_idx)
    match_tol = 0.99  # match at least 99%.
    if not (src.offsets == 0.0).all():
        match_tol = 0.89  # match at least 89% with offsets.
    np.testing.assert_array_less(match_tol, n_match / src.n)

    # Check dtype pass-through.
    assert cl.scls_scores.dtype == orient_est.dtype


def test_global_J_sync(orient_est):
    """
    For this test we build a set of relative rotations, Rijs, of shape
    (npairs, order(D2), 3, 3) and randomly J_conjugate them. We then test
    that the J-configuration is correctly detected and that J-synchronization
    is correct up to conjugation of the entire set.
    """
    # Grab set of rotations and generate a set of relative rotations, Rijs.
    rots = orient_est.src.rotations
    Rijs = np.zeros((orient_est.n_pairs, 4, 3, 3), dtype=orient_est.dtype)
    for p, (i, j) in enumerate(orient_est.pairs):
        Rij = rots[i].T @ orient_est.gs @ rots[j]
        np.random.shuffle(Rij)  # Mix up the ordering of Rijs
        Rijs[p] = Rij

    # J-conjugate a random set of Rijs.
    Rijs_conj = Rijs.copy()
    inds = np.random.choice(
        orient_est.n_pairs, size=orient_est.n_pairs // 2, replace=False
    )
    Rijs_conj[inds] = J_conjugate(Rijs[inds])

    # Create J-configuration conditions for the triplet Rij, Rjk, Rik.
    J_conds = {
        (False, False, False): 0,
        (True, True, True): 0,
        (True, False, False): 1,
        (False, True, True): 1,
        (False, True, False): 2,
        (True, False, True): 2,
        (False, False, True): 3,
        (True, True, False): 3,
    }

    # Construct ground truth J-configuration list based on `inds` of Rijs
    # that have been conjugated above.
    J_list_gt = np.zeros(len(orient_est.triplets), dtype=int)
    idx = 0
    for i, j, k in orient_est.triplets:
        ij = orient_est.pairs_to_linear[i, j]
        jk = orient_est.pairs_to_linear[j, k]
        ik = orient_est.pairs_to_linear[i, k]

        J_conf = (ij in inds, jk in inds, ik in inds)
        J_list_gt[idx] = J_conds[J_conf]
        idx += 1

    # Perform J-configuration and compare to ground truth.
    J_list = orient_est._J_configuration(Rijs_conj)
    np.testing.assert_equal(J_list, J_list_gt)

    # Perform global J-synchronization and check that
    # Rijs_sync is equal to either Rijs or J_conjugate(Rijs).
    Rijs_sync = orient_est._global_J_sync(Rijs_conj)
    need_to_conj_Rijs = not np.allclose(Rijs_sync[inds][0], Rijs[inds][0])
    if need_to_conj_Rijs:
        np.testing.assert_allclose(Rijs_sync, J_conjugate(Rijs))
    else:
        np.testing.assert_allclose(Rijs_sync, Rijs)

    # Check dtype pass-through.
    assert Rijs_sync.dtype == orient_est.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_global_J_sync_single_triplet(dtype):
    """
    This exercises the J-synchronization algorithm using the smallest
    possible problem size, a single triplets of relative rotations Rijs.
    """
    # Generate 3 image source and orientation object.
    src = Simulation(n=3, L=10, dtype=dtype, seed=SEED)
    orient_est = build_cl_from_source(src)

    # Grab set of rotations and generate a set of relative rotations, Rijs.
    rots = orient_est.src.rotations
    Rijs = np.zeros((orient_est.n_pairs, 4, 3, 3), dtype=orient_est.dtype)
    for p, (i, j) in enumerate(orient_est.pairs):
        Rij = rots[i].T @ orient_est.gs @ rots[j]
        np.random.shuffle(Rij)  # Mix up the ordering of Rijs
        Rijs[p] = Rij

    # J-conjugate a random Rij.
    Rijs_conj = Rijs.copy()
    inds = np.random.choice(orient_est.n_pairs, size=1, replace=False)
    Rijs_conj[inds] = J_conjugate(Rijs[inds])

    # Perform global J-synchronization and check that
    # Rijs_sync is equal to either Rijs or J_conjugate(Rijs).
    Rijs_sync = orient_est._global_J_sync(Rijs_conj)
    need_to_conj_Rijs = not np.allclose(Rijs_sync[inds][0], Rijs[inds][0])
    if need_to_conj_Rijs:
        np.testing.assert_allclose(Rijs_sync, J_conjugate(Rijs))
    else:
        np.testing.assert_allclose(Rijs_sync, Rijs)


def test_sync_colors(orient_est):
    """
    A set of estimated relative rotations, Rijs, have the shape (n_pairs, 4, 3, 3),
    where each 4-tuple Rij is given by Rij = Ri.T @ g_m @ Rj, for m in [0, 1, 2, 3],
    where each g_m is an element of the D2 symmetry group. The ordering of the symmetry
    group elements, g_m, is unknown and independent between Rijs. The `_sync_colors`
    algorithm forms the set of vijs of shape (n_pairs, 3, 3, 3), where each vij, given
    by vij = (Rij[0] + Rij[m]) / 2 with m = 1, 2, 3, is some permutation of the outer
    products of the k'th rows of the rotation matrices Ri and Rj, for k = 0, 1, 2.

    The 'sync_colors` algorithm uses a colored graph to partition the set of vijs
    based on k'th row outer products and returns those outer products along with
    a color mapping encoding a permutation for each vij.

    In this test we form a set of Rijs with randomly ordered symmetry group elements
    and extract the ground truth color permutations based on that ordering. We then
    construct a set of ground truth vijs adjusted by the ground truth color permuations.
    We then compare estimated vijs and color permutations to ground truth.
    """
    # Grab set of rotations and generate a set of relative rotations, Rijs.
    rots = orient_est.src.rotations
    Rijs = np.zeros((orient_est.n_pairs, 4, 3, 3), dtype=orient_est.dtype)
    gt_colors = np.zeros((orient_est.n_pairs, 3), dtype=int)

    with Random(123):
        for p, (i, j) in enumerate(orient_est.pairs):
            gs = orient_est.gs
            if p > 0:
                np.random.shuffle(gs)  # Mix up the ordering of all but 1st Rijs.

            # Compute the rotation row permutation created by the ordering of gs.
            # See Proposition 5.1 in the related publication for details.
            for m in range(3):
                gt_colors[p, m] = np.argmax(
                    np.sum(abs(0.5 * (gs[0] + gs[m + 1])), axis=0)
                )

            # Compute Rijs with shuffled gs.
            Rij = rots[i].T @ gs @ rots[j]
            Rijs[p] = Rij

    # Compute ground truth m'th row outer products.
    vijs = np.zeros((orient_est.n_pairs, 3, 3, 3), dtype=orient_est.dtype)
    for p, (i, j) in enumerate(orient_est.pairs):
        for m in range(3):
            row = gt_colors[p, m]
            vijs[p, m] = np.outer(rots[i][row], rots[j][row])

    # Perform color synchronization.
    # `est_vijs` is shape (n_pairs, 3, 3, 3) where est_vijs[ij, m] corresponds
    # to the outer product vij_m = rots[i, m].T @ rots[j, m] where m is the m'th row
    # of the rotations matrices Ri and Rj. `est_colors` partitions the set of `est_vijs`
    # such that the indices of `est_colors` corresponds to the row index m.
    est_colors, est_vijs = orient_est._sync_colors(Rijs)

    # Reshape `est_colors` to shape (n_pairs, 3) and use to index est_vijs into the
    # correctly order 3rd row outer products vijs.
    est_colors = est_colors.reshape(orient_est.n_pairs, 3)

    # `est_colors` is an arbitrary permutation (but globally consistent), and we know
    # that est_colors[0] should correspond to the ordering [0, 1, 2] due to the construction
    # of Rijs[0] using the symmetric rotations g0, g1, g2, g3 in non-permuted order.
    # So we sort the columns such that est_colors[0] = [0,1,2].

    # Create a mapping array
    perm = est_colors[0]
    mapping = np.zeros_like(perm)
    mapping[perm] = np.arange(3)

    # Apply this mapping to all rows of the est_colors array
    est_colors_mapped = mapping[est_colors]

    # Check that remapped color permutations match ground truth.
    np.testing.assert_allclose(est_colors_mapped, gt_colors)

    # est_vijs_synced should match the ground truth vijs up to the sign of each row.
    # So we multiply by the sign of the first column of the last two axes to sync signs.
    vijs = vijs * np.sign(vijs[..., 0])[..., None]
    est_vijs = est_vijs * np.sign(est_vijs[..., 0])[..., None]
    np.testing.assert_allclose(vijs, est_vijs, atol=utest_tolerance(orient_est.dtype))

    # Check dtype pass-through.
    assert est_vijs.dtype == orient_est.dtype


def test_sync_signs(orient_est):
    """
    Sign synchronization consumes a set of m'th row outer products along with
    a color synchronizing vector and returns a set of rotation matrices
    that are the result of synchronizing the signs of the rows of the outer
    products and factoring the outer products to form the rows of the rotations.

    In this test we provide a color-synchronized set of m'th row outer products
    with a corresponding color vector and test that the output rotations
    equivalent to the ground truth rotations up to a global alignment.
    """
    rots = orient_est.src.rotations

    # Compute ground truth m'th row outer products.
    vijs = np.zeros((orient_est.n_pairs, 3, 3, 3), dtype=orient_est.dtype)
    for p, (i, j) in enumerate(orient_est.pairs):
        for m in range(3):
            vijs[p, m] = np.outer(rots[i][m], rots[j][m])

    # We will pass in m'th row outer products that are color synchronized,
    # ie. colors = [0, 1, 2, 0, 1, 2, ...]
    perm = np.array([0, 1, 2])
    colors = np.tile(perm, orient_est.n_pairs)

    # Estimate rotations and check against ground truth.
    rots_est = orient_est._sync_signs(vijs, colors)
    mean_aligned_angular_distance(rots, rots_est, degree_tol=1e-5)

    # Check dtype pass-through.
    assert rots_est.dtype == orient_est.dtype


####################
# Helper Functions #
####################


def g_sync_d2(rots, rots_gt):
    """
    Every estimated rotation might be a version of the ground truth rotation
    rotated by g^{s_i}, where s_i = 0, 1, ..., order. This method synchronizes the
    ground truth rotations so that only a single global rotation need be applied
    to all estimates for error analysis.

    :param rots: Estimated rotation matrices
    :param rots_gt: Ground truth rotation matrices.

    :return: g-synchronized ground truth rotations.
    """
    assert len(rots) == len(
        rots_gt
    ), "Number of estimates not equal to number of references."
    n_img = len(rots)
    dtype = rots.dtype

    rots_symm = DnSymmetryGroup(2).matrices.astype(dtype, copy=False)
    order = len(rots_symm)

    A_g = np.zeros((n_img, n_img), dtype=complex)

    pairs = all_pairs(n_img)

    for i, j in pairs:
        Ri = rots[i]
        Rj = rots[j]
        Rij = Ri.T @ Rj

        Ri_gt = rots_gt[i]
        Rj_gt = rots_gt[j]

        diffs = np.zeros(order)
        for s, g_s in enumerate(rots_symm):
            Rij_gt = Ri_gt.T @ g_s @ Rj_gt
            diffs[s] = min(
                [
                    np.linalg.norm(Rij - Rij_gt),
                    np.linalg.norm(Rij - J_conjugate(Rij_gt)),
                ]
            )

        idx = np.argmin(diffs)

        A_g[i, j] = np.exp(-1j * 2 * np.pi / order * idx)

    # A_g(k,l) is exp(-j(-theta_k+theta_l))
    # Diagonal elements correspond to exp(-i*0) so put 1.
    # This is important only for verification purposes that spectrum is (K,0,0,0...,0).
    A_g += np.conj(A_g).T + np.eye(n_img)

    _, eig_vecs = np.linalg.eigh(A_g)
    leading_eig_vec = eig_vecs[:, -1]

    angles = np.exp(1j * 2 * np.pi / order * np.arange(order))
    rots_gt_sync = np.zeros((n_img, 3, 3), dtype=dtype)

    for i, rot_gt in enumerate(rots_gt):
        # Since the closest ccw or cw rotation are just as good,
        # we take the absolute value of the angle differences.
        angle_dists = np.abs(np.angle(leading_eig_vec[i] / angles))
        power_g_Ri = np.argmin(angle_dists)
        rots_gt_sync[i] = rots_symm[power_g_Ri] @ rot_gt

    return rots_gt_sync


def build_cl_from_source(source):
    # Search for common lines over less shifts for 0 offsets.
    max_shift = 0
    shift_step = 1
    if source.offsets.all() != 0:
        max_shift = 0.2
        shift_step = 0.02  # Reduce shift steps for non-integer offsets of Simulation.

    orient_est = CLSymmetryD2(
        source,
        max_shift=max_shift,
        shift_step=shift_step,
        n_theta=180,
        n_rad=source.L,
        grid_res=350,  # Tuned for speed
        inplane_res=12,  # Tuned for speed
        eq_min_dist=10,  # Tuned for speed
        epsilon=0.001,
        seed=SEED,
    )
    return orient_est
