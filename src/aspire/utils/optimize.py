import logging

import numpy as np

logger = logging.getLogger(__name__)


def fill_struct(obj=None, att_vals=None):
    """
    Fill object with attributes in a dictionary.

    If a struct is not given a new object will be created and filled.
    If the given struct has a field in att_vals, the original field will stay,
    unless specified otherwise in overwrite.
    att_vals is a dictionary with string keys, and for each key:
    if hasattr(s, key) and key in overwrite:
        pass
    else:
        setattr(s, key, att_vals[key])
    :param obj:
    :param att_vals:
    :param overwrite
    :return:
    """
    # TODO should consider making copy option - i.e that the input won't change

    if obj is None:
        obj = {}
    if att_vals is None:
        return obj

    for key in att_vals.keys():
        if key not in obj.keys():
            obj[key] = att_vals[key]

    return obj


def conj_grad(a_fun, b, cg_opt=None, init=None):
    """
    Conjugate Gradient method to solve the linear system.

    This is corresponding to the implemented version in the ASPIRE Matlab package.
    :param a_fun:  A function handle specifying the linear operation x -> Ax.
        When multiple right-hand sides are supplied, this function takes as
        input an array of shape (n, p), where n is the number of right-hand
        sides and p is the dimension of the space.
    :param b:  The vector consisting of the right hand side of Ax = b. Again,
        n different right-hand sides are given by supplying an array of shape
        (n, p).
    :param cg_opt: The parameters for the conjugate gradient method, including:
            max_iter: Maximum number of iterations (default 50).
            verbose: The extent to which information on progress should be
                output to the terminal (default 1).
            iter_callback: If non-empty, specifies a function to be called at
                the end of every iteration. In this case, iter_callback must be a
                function handle taking as single argument the info structure at
                the current iteration. For information on the info structure,
                see below (default []).
            preconditioner: If non-empty, specifies a preconditioner to be
                used in every iteration as a function handle defining the linear
                operator x -> Px (default []).
            rel_tolerance: The relative error at which to stop the algorithm,
                even if it has not yet reached the maximum number of iterations
                (default 1e-15).
            store_iterates: Defines whether to store each intermediate results
                in the info structure under the x, p and r fields. Since this
                may require a large amount of memory, this is not recommended
                (default false).
    :param init: A structure specifying the starting point of the algorithm.
            This can contain values of x or p that will be used for initialization
            (default empty).
    :return: The output result includes:
            x: The result of the conjugate gradient method after max_iter iterations
                or once the residual norm has decreased below rel_tolerance, relative.
            obj: The value of the objective function at the last iteration.
            info: A structure array containing intermediate information obtained
            during each iteration. These fields include:
            - iter: The iteration number.
            - x (for store_iterates true): The value of x.
            - r (for store_iterates true): The residual vector.
            - p (for store_iterates true): The p vector.
            - res: The square norm of the residual.
            - obj: The objective function.
    """

    def identity(input_x):
        return input_x

    default_opt = {
        "verbose": 0,
        "max_iter": 50,
        "iter_callback": [],
        "store_iterates": False,
        "rel_tolerance": 1e-15,
        "precision": b.dtype,
        "preconditioner": identity,
    }

    cg_opt = fill_struct(cg_opt, default_opt)

    default_init = {"x": None, "p": None}
    init = fill_struct(default_init, init)
    if init["x"] is None:
        x = np.zeros(b.shape, dtype=b.dtype)
    else:
        x = init["x"]

    b_norm = np.linalg.norm(b)
    r = b.copy()

    # Need the copy call to ensure that s and r are not identical in the case
    # of an identity preconditioner.
    s = cg_opt["preconditioner"](r.copy())

    if np.any(x != 0):
        if cg_opt["verbose"]:
            logger.info("[CG] Calculating initial residual")
        a_x = a_fun(x)
        r = r - a_x
        s = cg_opt["preconditioner"](r)
    else:
        a_x = np.zeros(x.shape, dtype=b.dtype)

    obj = np.real(np.sum(x.conj() * a_x, -1) - 2 * np.real(np.sum(np.conj(b * x), -1)))

    if init["p"] is None:
        p = s.copy()
    else:
        p = init["p"]

    info = fill_struct(att_vals={"iter": [0], "res": [np.linalg.norm(r)], "obj": [obj]})
    if cg_opt["store_iterates"]:
        info = fill_struct(info, att_vals={"x": [x], "r": [r], "p": [p]})

    if cg_opt["verbose"]:
        logger.info(
            "[CG] Initialized. Residual: {}. Objective: {}".format(
                np.linalg.norm(info["res"][0]), np.sum(info["obj"][0])
            )
        )

    if b_norm == 0:
        # Matlat code returns b_norm == 0, this break the Python code when b = 0
        return x, obj, info

    for i in range(1, cg_opt["max_iter"] + 1):
        if cg_opt["verbose"]:
            logger.info("[CG] Applying matrix & preconditioner")

        a_p = a_fun(p)
        old_gamma = np.real(np.sum(s.conj() * r, -1))
        alpha = old_gamma / np.real(np.sum(p.conj() * a_p, -1))
        x += alpha[..., np.newaxis] * p
        a_x += alpha[..., np.newaxis] * a_p

        r -= alpha[..., np.newaxis] * a_p
        s = cg_opt["preconditioner"](r.copy())
        new_gamma = np.real(np.sum(r.conj() * s, -1))
        beta = new_gamma / old_gamma
        p *= beta[..., np.newaxis]
        p += s

        obj = np.real(
            np.sum(x.conj() * a_x, -1) - 2 * np.real(np.sum(np.conj(b * x), -1))
        )
        res = np.sqrt(np.sum(r ** 2, -1))
        info["iter"].append(i)
        info["res"].append(res)
        info["obj"].append(obj)
        if cg_opt["store_iterates"]:
            info["x"].append(x)
            info["r"].append(r)
            info["p"].append(p)

        if cg_opt["verbose"]:
            logger.info(
                "[CG] Iteration {}. Residual: {}. Objective: {}".format(
                    i, np.linalg.norm(info["res"][i]), np.sum(info["obj"][i])
                )
            )

        if np.all(res < b_norm * cg_opt["rel_tolerance"]):
            break

    if i == cg_opt["max_iter"]:
        logger.warning("[CG] Conjugate gradient reached maximum number of iterations!")

    return x, obj, info
