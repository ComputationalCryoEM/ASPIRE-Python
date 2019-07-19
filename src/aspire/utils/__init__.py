def ensure(cond, error_message=None):
    """
    assert statements in Python are sometimes optimized away by the compiler, and are for internal testing purposes.
    For user-facing assertions, we use this simple wrapper to ensure conditions are met at relevant parts of the code.

    :param cond: Condition to be ensured
    :param error_message: An optional error message if condition is not met
    :return: If condition is met, returns nothing, otherwise raises AssertionError
    """
    if not cond:
        raise AssertionError(error_message)
