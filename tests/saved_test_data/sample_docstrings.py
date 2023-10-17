def good_fun1(frog, dog):
    """
    This docstring is properly formatted.

    It has a multi-line, multi-section body
    followed by exactly one blank line.

    :param frog: This param description is
        multiline.
    :param dog: Single line description
    :return: A frog on a dog
    """


def good_fun2():
    """
    This function has only a return.

    :return: Just a return.
    """


def good_fun3():
    def nested_fun(bip):
        """
        This is a properly formatted docstring
        in a nested function.

        :param bip: A small bip
        :return: A large bop
        """


def bad_fun1(cat, hat):
    """
    This docstring is missing a blank line
    between the body and parameter sections.
    :param cat: A cat.
    :param hat: A hat.
    :return: A cat in a hat.
    """


def bad_fun2(foo):
    """
    This docstring has too many blank lines between
    the body and parameter sections.


    :param foo: foo description.
    :return: bar
    """


def bad_fun3():
    def nested_fun(bip):
        """
        This is an improperly formatted docstring
        in a nested function.
        :param bip: A small bip
        :return: A large bop
        """
