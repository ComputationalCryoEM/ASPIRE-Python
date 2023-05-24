import warnings

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
        q = np.array([1, 0, 0, 0])
    else:
        q = np.array(
            [
                np.cos(v / 2),
                np.sin(v / 2) / v * u[0],
                np.sin(v / 2) / v * u[1],
                np.sin(v / 2) / v * u[2],
            ]
        )
    R = np.zeros((3, 3), dtype=np.float32)
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
    q = np.zeros(4)
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
        return np.zeros(3)
    else:
        return v / np.sin(v / 2) * q[1:4]


def q_to_rot(q):
    """
    What does this do?

    :param q: what is q
    :return: what does this return
    """
    R = np.zeros((3, 3))
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
    X = np.zeros(L)
    mid = int(L / 2)
    for i in range(L):
        X[i] = i - mid

    vx = np.sum(v, axis=(1, 2))
    vy = np.sum(v, axis=(0, 2))
    vz = np.sum(v, axis=(0, 1))
    m = np.array([X @ vx, X @ vy, X @ vz])
    vol_b = shift(vol, -m, order=order_shift, mode="constant")
    return vol_b


def align_BO(vol0, vol_given, para, reflect=False):
    """
    What does this do?

    :param vol0: ?
    :param vol_given: ?
    :param para: ?
    :param reflect: ?
    :return: what does this return
    """

    loss_type = para[0]
    ds = para[1]
    Niter = para[2]
    refine = para[3]

    vol0_ds = vol0.downsample(ds)
    vol_given_ds = vol_given.downsample(ds)
    if loss_type == "wemd":
        lengthscale = 0.75
        warnings.filterwarnings("ignore")
        wavelet = "sym3"
        level = 6
        embed_0 = wemd_embed(vol0_ds._data[0], wavelet, level)
    if loss_type == "eu":
        lengthscale = 1

    def loss(R):
        v_rot = vol_given_ds.rotate(Rotation(R))._data[0]
        if loss_type == "eu":
            return norm(vol0_ds - v_rot)
        if loss_type == "wemd":
            warnings.filterwarnings("ignore")
            embed_rot = wemd_embed(v_rot, wavelet, level)
            return norm(embed_rot - embed_0, ord=1)

    def cf(x1, x2):
        d = norm(x1 - x2) / lengthscale
        return np.exp(-(d**2) / 2)

    def cf_grad(x1, x2):
        return cf(x1, x2) * (x2 - x1) / lengthscale**2

    R = np.zeros((Niter, 3, 3), dtype="float32")
    ninit = 1
    R[0] = np.float32(np.eye(3))
    if reflect:
        manifold = pymanopt.manifolds.Stiefel(3, 3)
    else:
        manifold = pymanopt.manifolds.SpecialOrthogonalGroup(3)

    C = np.zeros((Niter, Niter))
    for i in range(ninit):
        for j in range(ninit):
            C[i, j] = cf(R[i], R[j])

    y = np.zeros(Niter)
    for i in range(ninit):
        y[i] = loss(R[i])

    tau = 1e-3
    max_iter = 500
    min_grad = 0.1
    min_sz = 0.1
    verb = 0

    for t in range(ninit, Niter):
        q = np.linalg.solve(C[:t, :t] + tau * np.eye(t), y[:t])

        @pymanopt.function.numpy(manifold)
        def cost(new):
            kx = np.array([cf(new, R[j]) for j in range(t)])
            mu = kx @ q
            return mu

        @pymanopt.function.numpy(manifold)
        def eu_grad(new):
            kx_grad = np.array([cf_grad(new, R[j]) for j in range(t)])
            grad = np.einsum("ijk,i", kx_grad, q)
            return grad

        problem = pymanopt.Problem(manifold, cost, euclidean_gradient=eu_grad)
        optimizer = pymanopt.optimizers.SteepestDescent(
            max_iterations=max_iter,
            min_gradient_norm=min_grad,
            min_step_size=min_sz,
            verbosity=verb,
        )
        result = optimizer.run(problem)
        R_new = np.float32(result.point)

        y[t] = loss(R_new)
        R[t] = R_new
        k = np.array([cf(R_new, R[i]) for i in range(t + 1)])
        C[t, 0:t] = k[0:t]
        C[0 : t + 1, t] = k

    idx = y.argmin(0)
    R_init = R[idx]

    if ds > 32:
        vol0_ds = vol0_ds.downsample(32)
        vol_given_ds = vol_given_ds.downsample(32)
    if refine:
        sign = np.sign(np.linalg.det(R_init))

        def loss_u(u):
            R = sign * u_to_rot(u)
            v_rot = vol_given_ds.rotate(Rotation(R))._data[0]
            return norm(vol0_ds - v_rot)

        x0 = rot_to_u(sign * R_init)
        result = minimize(loss_u, x0, method="nelder-mead", options={"disp": False})
        u_est = result.x
        R_est = sign * u_to_rot(u_est)

    else:
        R_est = R_init

    return R_init, R_est
