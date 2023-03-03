import abc
import logging

import numpy as np
import numpy.ma as ma
from scipy.stats.mstats import winsorize

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

        # When provided an image, keep track of shape.
        self._return_image_size = None

    @abc.abstractmethod
    def __call__(self, stack):
        """
        Stack the elements of `stack`.

        Stack admits an Image class, or an Numpy array.

        In the case of Numpy array, the data should be 2D
        where the first (slow) dimension is stack axis,
        and signal data is flattened to the last (fastest) axis.
        In this case an Numpy array is returned.

        When passing `Image`, the stack_shape must be 1D.
        In this case an `Image` is returned.
        Users with multidimensional `Image` data
        may use `Image.stack_reshape` or stack slicing
        before/after stacking.

        :param stack: Image instance (or Numpy array).

        :return: Stacked data as Image (or Numpy array).
        """

    def _check_and_convert(self, stack):
        """
        Check stack and returns consistent 2D np.array.

        :param stack: Image instance (or Numpy array).

        :return: 2D Numpy array.
        """

        # Careful, we need to reset this in case this stack is different from last.
        self._return_image_size = None

        # Flatten image data.
        if isinstance(stack, Image):
            # Store the image size for returning later
            self._return_image_size = (1,) + stack.shape[1:]

            # Sanity check dimensions
            if stack.stack_ndim != 1:
                raise ValueError(
                    f"`stack` shape of Image should be 1D for ImageStacking not {stack.stack_shape}."
                    "  Try Image.stack_reshape if needed."
                )
            stack = stack.asnumpy().reshape(stack.n_images, -1)

        # By this point we should alway be an array with signal data in the last axis.
        if not isinstance(stack, np.ndarray):
            raise ValueError("`stack` should be `Image` instance or Numpy array.")
        elif stack.ndim != 2:
            raise ValueError(
                f"`stack` numpy array shape should be 2D for ImageStacker not {stack.shape}."
                "  Try Image.stack_reshape if needed."
            )

        return stack

    def _return(self, result):
        """
        When ImageStacker has been passed an `Image`,
        this method creates an `Image` instance to be returned.

        Stacking Numpy array will pass through.

        :param result: Result data as Numpy Array.

        :return: Image(result) or Numpy array(result) based on initial `stack` input type.
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
    """Stack using Sigma Rejection.

     When no outliers exist, sigma rejection should return equivalent
    of `mean`.  In the presence of outliers, pixels outside of
    `rejection_sigma` from the per-pixel-mean are discarded.

     For potentially less Gaussian distributions, 'FWHM' and 'FWTM'
     methods are provided.  These will take the mean of values lieing
     in the FWHM and FWTM. Essentially this is the same wing clipping
     procedure.

     Note, in both cases, user's are responsible for ensuring metohds
     are called on reasonable data (in FW* cases we should be talking
     intesnities). No corrections or pedestals are incorporated at this
     time."""

    _width_methods = {"FWHM": 0.5, "FWTM": 0.1}

    def __init__(self, rejection_sigma=3):
        """
        Instantiates SigmaRejectionImageStacker instance with
        presribed `rejection_sigma`.

        :param rejection_sigma: Values falling outside
            `rejection_sigma` standard deviations are
            rejected. Defaults to 3.  Also accepts 'FWHM' and 'FWTM'
            corresponding to per-pixel full width at half maximum and
            tenth maximum respectively.

        """
        # Handle string `rejection_sigma`
        self._method = self._gaussian
        if isinstance(rejection_sigma, str):
            self.rejection_sigma = rejection_sigma.upper()
            self._method = self._width
            if self.rejection_sigma not in self._width_methods:
                raise ValueError(
                    f"`rejection_sigma` must be numeric or {self._width_methods.keys()}."
                )
        else:
            self.sigma = float(rejection_sigma)

    def __call__(self, stack):
        """
        Dispatches rejection method based on `rejection_sigma`.
        """
        stack = self._check_and_convert(stack)
        return self._return(self._method(stack))

    def _gaussian(self, stack):
        """
        Gaussian rejection.
        """

        # Compute the mean and standard deviations, pixelwise
        means = stack.mean(axis=0)
        std_devs = stack.std(axis=0)

        # Compute values that lie outside sigma deviations
        outliers = np.abs(stack - means) > self.sigma * std_devs

        # Mask off the outliers
        masked_stack = ma.masked_array(stack, mask=outliers)

        # Return mean without the outliers
        return masked_stack.mean(axis=0)

    def _width(self, stack):
        """
        Width rejection.
        """
        # Compute per-pixel max
        maxima = stack.max(axis=0)

        # Find the per-pixel value for clipping
        y = maxima * self._width_methods[self.rejection_sigma]

        # Compute outliers, values below clipping value.
        outliers = stack < y

        # Mask off the outliers
        masked_stack = ma.masked_array(stack, mask=outliers)

        # Return mean withou the outliers
        return masked_stack.mean(axis=0)


class WinsorizedImageStacker(ImageStacker):
    """
    Stack using Winsorization.

    Winsorizing is similar to SigmaRejectionImageStacker,
    excect it admits a `percentile` and replaces rejected
    values with values at +/- `percentile`.
    """

    def __init__(self, percentile=0.1):
        """Instantiates WinsorizedImageStacker instance with
        presribed `percentile`.

        See scipy docs for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

        :param percentile: Float or tuple of floats.  Values must be
            [0,1] and represent the percentile used for trimming.  A
            tuple allows different lower and upper percentiles
            respectively. Example (0.1, 0.2) would Winsorize the lower
            10% and upper 20% of values.  In the case of a single
            value, that value will be used for both upper and lower
            percentiles.
        """
        # Convert scalars to tuples.
        if not isinstance(percentile, tuple):
            percentile = (float(percentile),) * 2
        if (min(percentile) < 0) or (max(percentile) > 1):
            raise ValueError(f"`percentile` must be [0,1], passed {percentile}.")
        self.percentile = percentile

    def __call__(self, stack):
        """
        Winsorizing
        """
        stack = self._check_and_convert(stack)

        stack = winsorize(stack, limits=self.percentile, inplace=False)

        # Return mean of Winsorized data.
        return self._return(stack.mean(axis=0))


class PoissonRejectionImageStacker(ImageStacker):
    """
    Stack using Poisson Rejection.
    """


class RobustChauvenetRejectionImageStacker(ImageStacker):
    """
    Stack using Robust Chauvenet Outlier Rejection.
    """
