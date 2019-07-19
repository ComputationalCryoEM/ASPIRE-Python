import numpy as np


def fill_struct(obj=None, att_vals=None, overwrite=None):
    """
    Fill object with attributes in a dictionary.
    If a struct is not given a new object will be created and filled.
    If the given struct has a field in att_vals, the original field will stay, unless specified otherwise in overwrite.
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
        class DisposableObject:
            pass

        obj = DisposableObject()

    if att_vals is None:
        return obj

    if overwrite is None or not overwrite:
        overwrite = []
    if overwrite is True:
        overwrite = list(att_vals.keys())

    for key in att_vals.keys():
        if hasattr(obj, key) and key not in overwrite:
            continue
        else:
            setattr(obj, key, att_vals[key])

    return obj


def conj_grad(a_fun, b, cg_opt=None, init=None):

    def identity(input_x):
        return input_x

    cg_opt = fill_struct(cg_opt, {'max_iter': 50, 'verbose': 0, 'iter_callback': [], 'preconditioner': identity,
                                  'rel_tolerance': 1e-15, 'store_iterates': False})
    init = fill_struct(init, {'x': None, 'p': None})
    if init.x is None:
        x = np.zeros(b.shape)
    else:
        x = init.x

    b_norm = np.linalg.norm(b)
    r = b.copy()
    s = cg_opt.preconditioner(r)

    if np.any(x != 0):
        if cg_opt.verbose:
            print('[CG] Calculating initial residual')
        a_x = a_fun(x)
        r = r-a_x
        s = cg_opt.preconditioner(r)
    else:
        a_x = np.zeros(x.shape)

    obj = np.real(np.sum(x.conj() * a_x, 0) - 2 * np.real(np.sum(np.conj(b * x), 0)))

    if init.p is None:
        p = s
    else:
        p = init.p

    info = fill_struct(att_vals={'iter': [0], 'res': [np.linalg.norm(r)], 'obj': [obj]})
    if cg_opt.store_iterates:
        info = fill_struct(info, att_vals={'x': [x], 'r': [r], 'p': [p]})

    if cg_opt.verbose:
        print('[CG] Initialized. Residual: {}. Objective: {}'.format(np.linalg.norm(info.res[0]), np.sum(info.obj[0])))

    if b_norm == 0:
        print('b_norm == 0')
        return

    i = 0
    for i in range(1, cg_opt.max_iter):
        if cg_opt.verbose:
            print('[CG] Applying matrix & preconditioner')

        a_p = a_fun(p)
        old_gamma = np.real(np.sum(s.conj() * r))

        alpha = old_gamma / np.real(np.sum(p.conj() * a_p))
        x += alpha * p
        a_x += alpha * a_p

        r -= alpha * a_p
        s = cg_opt.preconditioner(r)
        new_gamma = np.real(np.sum(r.conj() * s))
        beta = new_gamma / old_gamma
        p *= beta
        p += s

        obj = np.real(np.sum(x.conj() * a_x, 0) - 2 * np.real(np.sum(np.conj(b * x), 0)))
        res = np.linalg.norm(r)
        info.iter.append(i)
        info.res.append(res)
        info.obj.append(obj)
        if cg_opt.store_iterates:
            info.x.append(x)
            info.r.append(r)
            info.p.append(p)

        if cg_opt.verbose:
            print('[CG] Initialized. Residual: {}. Objective: {}'.format(np.linalg.norm(info.res[0]), np.sum(info.obj[0])))

        if np.all(res < b_norm * cg_opt.rel_tolerance):
            break

    # if i == cg_opt.max_iter - 1:
    #     raise Warning('Conjugate gradient reached maximum number of iterations!')
    return x, obj, info

