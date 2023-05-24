import numpy as np
import pymanopt
from numpy import pi
from numpy.linalg import norm
from scipy.ndimage import shift
from scipy.optimize import minimize

from aspire.operators import wemd_embed
from aspire.utils.rotation import Rotation


def u_to_rot(u):
    """
    What does this do?

    :param u: what is u
    :return: what does this return
    """

    v = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    if v == 0:
        q = np.array([1, 0, 0, 0], dtype=u.dtype)
    else:
        q = np.array(
            [
                np.cos(v / 2),
                np.sin(v / 2) / v * u[0],
                np.sin(v / 2) / v * u[1],
                np.sin(v / 2) / v * u[2],
            ]
        )
    R = np.zeros((3, 3), dtype=u.dtype)
    R[0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    R[0, 1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    R[0, 2] = 2 * q[0] * q[2] + 2 * q[1] * q[3]
    R[1, 0] = 2 * q[1] * q[2] + 2 * q[0] * q[3]
    R[1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    R[1, 2] = -2 * q[0] * q[1] + 2 * q[2] * q[3]
    R[2, 0] = -2 * q[0] * q[2] + 2 * q[1] * q[3]
    R[2, 1] = 2 * q[0] * q[1] + 2 * q[2] * q[3]
    R[2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    return R


def rot_to_u(R):
    """
    What does this do?

    :param R: what is R
    :return: what does this return
    """
    q = np.zeros(4, dtype=R.dtype)
    if R[1, 1] > -R[2, 2] and R[0, 0] > -R[1, 1] and R[0, 0] > -R[2, 2]:
        q[0] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        q[1] = (R[2, 1] - R[1, 2]) / q[0]
        q[2] = (R[0, 2] - R[2, 0]) / q[0]
        q[3] = (R[1, 0] - R[0, 1]) / q[0]

    elif R[1, 1] < -R[2, 2] and R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        q[1] = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / q[1]
        q[2] = (R[1, 0] + R[0, 1]) / q[1]
        q[3] = (R[2, 0] + R[0, 2]) / q[1]

    elif R[1, 1] > R[2, 2] and R[0, 0] < R[1, 1] and R[0, 0] < -R[2, 2]:
        q[2] = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / q[2]
        q[1] = (R[1, 0] + R[0, 1]) / q[2]
        q[3] = (R[2, 1] + R[1, 2]) / q[2]

    elif R[1, 1] < R[2, 2] and R[0, 0] < -R[1, 1] and R[0, 0] < R[2, 2]:
        q[3] = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
        q[0] = (R[1, 0] - R[0, 1]) / q[3]
        q[1] = (R[2, 0] + R[0, 2]) / q[3]
        q[2] = (R[2, 1] + R[1, 2]) / q[3]

    q = q / 2
    v = 2 * np.arccos(q[0])
    if v == 0:
        return np.zeros(3, R.dtype)
    else:
        return v / np.sin(v / 2) * q[1:4]


def q_to_rot(q):
    """
    What does this do?

    :param q: what is q
    :return: what does this return
    """
    R = np.zeros((3, 3), q.dtype)
    R[0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    R[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    R[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
    R[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    R[1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    R[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    R[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    R[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    R[2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    return R


def center(vol, order_shift, threshold=-np.inf):
    """
    What does this do?

    :param vol: ?
    :param order_shift: ?
    :param threshold: ?
    :return: what does this return
    """

    v = np.copy(vol)
    v.setflags(write=1)
    v[v < threshold] = 0
    v = v / v.sum()
    L = vol.shape[1]
    X = np.zeros(L, dtype=X.dtype)
    mid = int(L / 2)
    for i in range(L):
        X[i] = i - mid

    vx = np.sum(v, axis=(1, 2))
    vy = np.sum(v, axis=(0, 2))
    vz = np.sum(v, axis=(0, 1))
    m = np.array([X @ vx, X @ vy, X @ vz])
    vol_b = shift(vol, -m, order=order_shift, mode="constant")
    return vol_b


def align_BO(
    vol_ref,
    vol_given,
    loss_type="wemd",
    downsampling_level=32,
    max_iters=200,
    refine=True,
    reflect=False,
    ninit=1,
    tau=1e-3,
    man_max_iter=500,
    man_min_grad=0.1,
    man_min_sz=0.1,
    verbosity=0,
    dtype=None,
):
    """
    What does this do?

    :param vol_ref: ?
    :param vol_given: ?
    :param loss_type: 'wemd' or 'eu'. Default 'wemd'.
    :param downsampling_level: Downsampling (pixels)
    :param max_iters: Maximum iterations
    :param refine: Boolean, defaults True.
    :param reflect: ?
    :param ninit: ? 1
    :param tau: ? 1e-3
    :param max_iter: 500
    :param min_grad: ? 0.1
    :param min_sz: ?  0.1
    :param verbosity: ? 0
    :param dtype: Numeric dtype to perform computations with.
        Default `None` infers dtype from `vol_ref`.
    :return: what does this return
    """
    # Infer dtype
    dtype = np.dtype(dtype or vol_ref.dtype)

    # Convert volume data dtype if needed.
    vol_ref = vol_ref.astype(dtype, copy=False)
    vol_given = vol_given.astype(dtype, copy=False)

    # Store parameters specific to each loss_type.
    LOSS_TYPES = {
        "wemd": dict(lengthscale=0.75, wavelet="sym3", level=6),
        "eu": dict(lengthscale=1),
    }
    loss_type = loss_type.lower()
    if loss_type not in LOSS_TYPES.keys():
        raise ValueError(f"BOTalign `loss_type` must be one of {LOSS_TYPES.keys()}")

    # Lookup our params
    loss_params = LOSS_TYPES[loss_type]

    # Downsample Volumes
    vol_ref_ds = vol_ref.downsample(downsampling_level)
    vol_given_ds = vol_given.downsample(downsampling_level)

    if loss_type == "wemd":
        embed_0 = wemd_embed(
            vol_ref_ds.asnumpy()[0], loss_params["wavelet"], loss_params["level"]
        )

    def loss(R):
        v = vol_given_ds.rotate(Rotation(R)).asnumpy()[0]
        if loss_type == "eu":
            return norm(v - vol_ref_ds)
        elif loss_type == "wemd":
            embed_v = wemd_embed(v, loss_params["wavelet"], loss_params["level"])
            return norm(embed_v - embed_0, ord=1)

    def cf(x1, x2):
        d = norm(x1 - x2) / loss_params["lengthscale"]
        return np.exp(-(d**2) / 2, dtype=dtype)

    def cf_grad(x1, x2):
        return cf(x1, x2) * (x2 - x1) / loss_params["lengthscale"] ** 2

    R = np.zeros((max_iters, 3, 3), dtype=dtype)
    R[0] = np.eye(3, dtype=dtype)
    if reflect:
        manifold = pymanopt.manifolds.Stiefel(3, 3)
    else:
        manifold = pymanopt.manifolds.SpecialOrthogonalGroup(3)

    C = np.zeros((max_iters, max_iters), dtype=dtype)
    for i in range(ninit):
        for j in range(ninit):
            C[i, j] = cf(R[i], R[j])

    y = np.zeros(max_iters, dtype=dtype)
    for i in range(ninit):
        y[i] = loss(R[i])

    for t in range(ninit, max_iters):
        q = np.linalg.solve(C[:t, :t] + tau * np.eye(t, dtype=dtype), y[:t])

        @pymanopt.function.numpy(manifold)
        def cost(new):
            kx = np.array([cf(new.astype(dtype, copy=False), R[j]) for j in range(t)])
            mu = kx @ q
            return mu

        @pymanopt.function.numpy(manifold)
        def eu_grad(new):
            kx_grad = np.array(
                [cf_grad(new.astype(dtype, copy=False), R[j]) for j in range(t)]
            )
            grad = np.einsum("ijk,i", kx_grad, q)
            return grad

        problem = pymanopt.Problem(manifold, cost, euclidean_gradient=eu_grad)
        optimizer = pymanopt.optimizers.SteepestDescent(
            max_iterations=man_max_iter,
            min_gradient_norm=man_min_grad,
            min_step_size=man_min_sz,
            verbosity=verbosity,
        )
        result = optimizer.run(problem)
        R_new = result.point.astype(dtype, copy=False)

        y[t] = loss(R_new)
        R[t] = R_new
        k = np.array([cf(R_new, R[i]) for i in range(t + 1)])
        C[t, 0:t] = k[0:t]
        C[0 : t + 1, t] = k

    idx = y.argmin(0)
    R_init = R[idx]

    if downsampling_level > 32:
        vol_ref_ds = vol_ref_ds.downsample(32)
        vol_given_ds = vol_given_ds.downsample(32)
    if refine:
        sign = np.sign(np.linalg.det(R_init))

        def loss_u(u):
            u = u.astype(dtype, copy=False)
            R = sign * u_to_rot(u)
            v_rot = vol_given_ds.rotate(Rotation(R)).asnumpy()[0]
            return norm(vol_ref_ds - v_rot)

        x0 = rot_to_u(sign * R_init)
        result = minimize(loss_u, x0, method="nelder-mead", options={"disp": False})
        u_est = result.x
        R_est = sign * u_to_rot(u_est)

    else:
        R_est = R_init

    return R_init, R_est
