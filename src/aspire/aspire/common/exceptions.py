class AspireException(Exception):
    pass


class WrongInput(AspireException):
    pass


class DimensionsIncompatible(AspireException):
    pass


class ErrorTooBig(AspireException):
    pass


class UnknownFormat(AspireException):
    pass
