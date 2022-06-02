import dill


def __run_encoded(payload):
    """
    Private method to decode and execute payload
    """

    # decode payload
    f, x = dill.loads(payload)
    # exec
    return f(*x)


def apply_async(mp_pool, function, args):
    """
    This method packs most arbitrary functions and their arguments
    into a payload that can be unpacked and run asynchronously
    in a multiprocessing pool.

    :param mp_pool: Multiprocessing Pool
    (see: https://docs.python.org/3/library/multiprocessing.html)
    :param function: Function to run async in the `pool`
    :param args: Arguments passed to `function`
    """

    # encode payload
    payload = dill.dumps((function, args))

    # (async)
    return mp_pool.apply_async(__run_encoded, (payload,))
