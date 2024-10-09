"""
This code is an ASPIRE port of Ruiyi Yang's `Bayesian Optimal Transport Alignment`.

`https://github.com/RuiyiYang/BOTalign`

`https://arxiv.org/pdf/2305.12310.pdf`
"""

import numpy as np
import pymanopt
from numpy.linalg import norm
from scipy.optimize import minimize

from aspire.utils.rotation import Rotation

# Store parameters specific to each loss_type.
# `lengthscale` is used to scale the modeled covariance
# functions, corresponding to equations 7 through 9 in the paper.
# `wavelet` and `level` are passed directly to `wemd_embed`.
loss_types = {
    "wemd": dict(lengthscale=0.75, wavelet="sym3", level=6),
    "euclidean": dict(lengthscale=1),
}


def align_BO(
    vol_ref,
    vol_given,
    loss_type="wemd",
    loss_params=None,
    downsampled_size=32,
    refinement_downsampled_size=32,
    max_iters=200,
    refine=True,
    tau=1e-3,
    surrogate_max_iter=500,
    surrogate_min_grad=0.1,
    surrogate_min_step=0.1,
    verbosity=0,
    dtype=None,
):
    """
    This function returns a rotation matrix R that best aligns vol_ref with the rotated version of vol_given.

    :param vol_ref: The reference `Volume`
    :param vol_given: The `Volume` to be aligned
    :param loss_type: 'wemd' or 'euclidean'. Default 'wemd'.
        If the heterogeneity between vol_ref and vol_given is high, 'euclidean' is recommended.
    :param loss_params: Optional dictionary for overriding parameters in `aspire.utils.bot_align.loss_types`.
        Defaults to empty dictionary.
    :param downsampled_size: Downsampling (pixels). Integer, defaults to 32.
        If alignment fails try larger values.
    :param refinement_downsampled_size: Downsampling (pixels) used with refinement. Integer, defaults to 32.
    :param max_iters: Maximum iterations. Integer, defaults 200.
        If alignment fails try larger values.
    :param refine: Whether to perform refinement. Boolean, defaults True.
    :param tau: Regularization parameter for surrogate problems. Numeric, defaults 1e-3.
    :param surrogate_max_iter: Stopping criterion for surrogate problems--maximum iterations. Interger, defaults 500.
    :param surrogate_min_grad: Stopping criterion for surrogate problems--minimum gradient norm. Numeric, defaults 0.1.
    :param surrogate_min_step: Stopping criterion for surrogate problems--minimum step size. Numeric, defaults 0.1.
    :param verbosity: Surrogate problem optimization detail level. integer, defaults 0 (silent). 2 is most verbose.
    :param dtype: Numeric dtype to perform computations with.
        Default `None` infers dtype from `vol_ref`.
    :return: Rotation matrix R_init (without refinement) or (R_init, R_est) (with refinement).
    """
    # Avoid utils/operators/utils circular import
    from aspire.operators import wemd_embed

    # Infer dtype
    dtype = np.dtype(dtype or vol_ref.dtype)

    # Raise error on volume dtype mismatch.
    if vol_ref.dtype != dtype:
        raise ValueError(f"`vol_ref` dtype {vol_ref.dtype} != {dtype}.")
    if vol_given.dtype != dtype:
        raise ValueError(f"`vol_ref` dtype {vol_given.dtype} != {dtype}.")

    loss_type = loss_type.lower()
    if loss_type not in loss_types.keys():
        raise ValueError(f"BOTalign `loss_type` must be one of {loss_types.keys()}")

    # Update defaults param items with any user specified items.
    if loss_params is None:
        loss_params = dict()
    loss_params = {**loss_types[loss_type], **loss_params}

    # Downsample Volumes
    vol_ref_ds = vol_ref.downsample(downsampled_size)
    vol_given_ds = vol_given.downsample(downsampled_size)

    if loss_type == "wemd":
        embed_0 = wemd_embed(
            vol_ref_ds.asnumpy()[0], loss_params["wavelet"], loss_params["level"]
        )

    def loss_fun(R):
        """
        Alignment loss function.
        """
        v = vol_given_ds.rotate(Rotation(R)).asnumpy()[0]
        if loss_type == "euclidean":
            return norm(v - vol_ref_ds)
        elif loss_type == "wemd":
            embed_v = wemd_embed(v, loss_params["wavelet"], loss_params["level"])
            return norm(embed_v - embed_0, ord=1)

    def cov_fun(x1, x2):
        """
        The squared exponential covariance function over SO(3).
        """
        d = norm(x1 - x2) / loss_params["lengthscale"]
        return np.exp(-(d**2) / 2, dtype=dtype)

    def cov_fun_grad(x1, x2):
        """
        Surrogate gradient of the squared exponential covariance function,
        with respect to `x1`.
        """
        # Corresponds to equation 11 in the paper.
        return cov_fun(x1, x2) * (x2 - x1) / loss_params["lengthscale"] ** 2

    R = np.zeros((max_iters, 3, 3), dtype=dtype)
    R[0] = np.eye(3, dtype=dtype)
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(3)

    cov = np.zeros((max_iters, max_iters), dtype=dtype)
    cov[0, 0] = cov_fun(R[0], R[0])

    loss = np.zeros(max_iters, dtype=dtype)
    loss[0] = loss_fun(R[0])

    for t in range(1, max_iters):
        # See discussion 3.2 and equation 10 in the paper.
        q = np.linalg.solve(cov[:t, :t] + tau * np.eye(t, dtype=dtype), loss[:t])

        @pymanopt.function.numpy(manifold)
        def cost(new, t=t, q=q):
            """
            Loss function for surrogate problems.
            """
            kx = np.array(
                [cov_fun(new.astype(dtype, copy=False), R[j]) for j in range(t)]
            )
            mu = kx @ q
            return mu

        @pymanopt.function.numpy(manifold)
        def euclidean_grad(new, t=t, q=q):
            """
            Gradient of the loss function for surrogate problems.
            """
            # Corresponds to equation 11 in the paper.
            kx_grad = np.array(
                [cov_fun_grad(new.astype(dtype, copy=False), R[j]) for j in range(t)]
            )

            kx_grad = kx_grad.reshape((t, 9))
            grad = q @ kx_grad
            grad = grad.reshape((3, 3))

            return grad

        problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_grad)
        optimizer = pymanopt.optimizers.SteepestDescent(
            max_iterations=surrogate_max_iter,
            min_gradient_norm=surrogate_min_grad,
            min_step_size=surrogate_min_step,
            verbosity=verbosity,
        )
        result = optimizer.run(problem)
        R_new = result.point.astype(dtype, copy=False)

        loss[t] = loss_fun(R_new)
        R[t] = R_new
        k = np.array([cov_fun(R_new, R[i]) for i in range(t + 1)])
        cov[t, 0:t] = k[0:t]
        cov[0 : t + 1, t] = k

    idx = loss.argmin(0)
    R_init = R[idx]

    if downsampled_size > refinement_downsampled_size:
        vol_ref_ds = vol_ref_ds.downsample(refinement_downsampled_size)
        vol_given_ds = vol_given_ds.downsample(refinement_downsampled_size)

    if refine:
        sign = np.sign(np.linalg.det(R_init))

        def loss_u(u):
            """
            Convert the alignment loss function over SO(3) to be over R^3.
            """
            u = u.astype(dtype, copy=False)
            R = sign * Rotation.from_rotvec(u).matrices[0]
            v_rot = vol_given_ds.rotate(Rotation(R)).asnumpy()[0]
            return norm(vol_ref_ds - v_rot)

        x0 = Rotation(sign * R_init).as_rotvec()[0]
        result = minimize(loss_u, x0, method="nelder-mead", options={"disp": False})
        u_est = result.x
        R_est = sign * Rotation.from_rotvec(u_est).matrices[0]

        return R_init, R_est
    else:
        return R_init
