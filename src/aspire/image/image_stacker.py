import abc
import logging
from warnings import catch_warnings, filterwarnings, warn

import numpy as np
import numpy.ma as ma

from aspire.image import Image

logger = logging.getLogger(__name__)


class ImageStacker(abc.ABC):
    """
    Interface for image and coefficient stacking classes.

    It is assumed that provided images are already aligned.
    Instances of `Stacker` act like functions.

    stacker = ImageStacker(...)
    stacker(images_to_stack).
    """

    def __init__(self):
        """
        Initialize ImageStacker instance.
        """

        self._return_image_size = None

    @abc.abstractmethod
    def __call__(self, stack):
        """
        Stack the elements of `stack`.

        Stack admits an Image class, or an ndarray.

        In the case of ndarray, the data should be 2D
        where the first (slow) dimension is stack axis,
        and data is flattened to the last (fastest) axis.
        In this case an Numpy array is returned.

        When passing `Image`, the stack_shape must be 1D.
        In this case an `Image` is returned.

        :param stack: Image instance or ndarray.
        """

    def _check_and_convert(self, stack):
        """
        Check stack and returns consistent 2D np.array.
        """

        self._return_image_size = None
        # If we are an image, flatten image data.
        if isinstance(stack, Image):
            # Store the image size for returning later
            self._return_image_size = (1,) + stack.shape[1:]

            if stack.stack_ndim != 1:
                raise ValueError(
                    f"`stack` shape of Image should be 1D for ImageStacking not {stack.stack_shape}."
                    "  Try Image.stack_reshape if needed."
                )
            stack = stack.asnumpy().reshape(stack.n_images, -1)

        # By this point we should alway be an np array with data in the last axis.
        if not isinstance(stack, np.ndarray):
            raise ValueError("`stack` should be `Image` instance or Numpy array.")
        elif stack.ndim != 2:
            raise ValueError(
                f"`stack` numpy array shape should be 2D for ImageStacker not {stack.shape}."
                "  Try Image.stack_reshape if needed."
            )

        # Store the stack shape.
        self._stack_shape = stack.shape[0]

        return stack

    def _return(self, result):
        """
        When ImageStacker has been passed an `Image`,
        this method creates an `Image` instance to be returned.

        :param result: Result data as Numpy Array.
        :return: Image(result) or Numpy array(result) based on initial call type.
        """
        if self._return_image_size is not None:
            result = Image(result.reshape(self._return_image_size))
        return result


class MeanImageStacker(ImageStacker):
    """
    Stack using `mean`.
    """

    def __call__(self, stack):
        stack = self._check_and_convert(stack)

        return self._return(stack.mean(axis=0))


class MedianImageStacker(ImageStacker):
    """
    Stack using `median`.
    """

    def __call__(self, stack):
        stack = self._check_and_convert(stack)

        return self._return(np.median(stack, axis=0))


class SigmaRejectionImageStacker(ImageStacker):
    """
    Stack using Sigma Rejection.
    """

    def __init__(self, rejection_sigma=2):
        """
        Instantiates SigmaRejectionImageStacker instance with
        presribed `rejection_sigma`.

        If no outliers should return equivalent of `mean`.
        In the presence of outliers, pixels outside of
        `rejection_sigma` from the per-pixel-mean are discarded.

        :param rejection_sigma: Values outside `rejection_sigma`
        """
        self.sigma = float(rejection_sigma)

    def __call__(self, stack):
        stack = self._check_and_convert(stack)

        # Compute the mean and standard deviations, pixelwise
        means = stack.mean(axis=0)
        std_deviations = stack.std(axis=0)

        # Compute values that lie outside sigma deviations
        outliers = np.abs(stack - means) > self.sigma * std_deviations

        # Mask off the outliers
        masked_stack = ma.masked_array(stack, mask=outliers)

        # Return mean withou the outliers
        return self._return(masked_stack.mean(axis=0))


class PoissonRejectionImageStacker(ImageStacker):
    """
    Stack using Poisson Rejection.
    """


class WinsorizedImageStacker(ImageStacker):
    """
    Stack using Winsorized Sigma Rejection.
    """


class RobustChauvenetRejectionImageStacker(ImageStacker):
    """
    Stack using Robust Chauvenet Outlier Rejection.
    """
