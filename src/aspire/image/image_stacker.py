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

        # This variable will be used to keep track of shape.
        self._return_image_size = None

    @abc.abstractmethod
    def _call(self, stack):
        """

        Subclasses must implement this method.

        Given a 2D Numpy array performs the stacking computation
        returning a 1D array. Packing and unpacking the array is done
        by `_check_and_convert` and `_return` respectively in
        `__call__`.

        :param stack: 2D Numpy array.

        :return: 1D Numpy array.
        """

    def __call__(self, stack):
        """
        Stack the elements of `stack`.

        Admits an Image class, or an Numpy array.

        In the case of Numpy array, the data should be 2D
        where the first (slow) dimension is stack axis,
        and signal data is flattened to the last (fastest) axis.
        In this case an Numpy array is returned.
        This allows performing stacking arithmetic in an arbitrary
        basis.

        When passing `Image`, the stack_shape must be 1D.
        In this case an `Image` is returned.
        Users with multidimensional `Image` data
        may use `Image.stack_reshape` or stack slicing
        before/after stacking to ensure the correct
        aggregation is occurring.

        If multidimensional stacking would be useful,
        submit a feature request.

        :param stack: Image instance (or Numpy array).

        :return: Stacked data as Image (or Numpy array).
        """
        stack = self._check_and_convert(stack)

        return self._return(self._call(stack))

    def _check_and_convert(self, stack):
        """
        Check stack and returns consistent 2D np.array.

        :param stack: Image instance (or Numpy array).

        :return: 2D Numpy array.
        """

        # Careful, we need to reset this in case this stack is
        # different from last call.
        self._return_image_size = None

        # Flatten image data.
        if isinstance(stack, Image):
            # Store the image size for returning later.
            self._return_image_size = (1,) + stack.shape[1:]

            # Sanity check dimensions.
            if stack.stack_ndim != 1:
                raise ValueError(
                    f"`stack` shape of Image should be 1D for ImageStacking not {stack.stack_shape}."
                    "  Try Image.stack_reshape if needed."
                )
            stack = stack.asnumpy().reshape(stack.n_images, -1)

        # By this point we should always be an array with signal data
        # in the last axis.
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

        In the case we started with a Numpy array this should pass
        through.

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

    def _call(self, stack):
        return stack.mean(axis=0)


class MedianImageStacker(ImageStacker):
    """
    Stack using `median`.
    """

    def _call(self, stack):
        return np.median(stack, axis=0)


class SigmaRejectionImageStacker(ImageStacker):
    """
    Stack using Sigma Rejection.

    When no outliers exist, sigma rejection is equivalent to `mean`.
    In the presence of outliers, pixels outside of `rejection_sigma`
    from the per-pixel-mean are discarded before the mean is
    performed.

    For potentially less Gaussian distributions, 'FWHM' and 'FWTM'
    methods are also provided.  These will take the mean of all values
    laying above half the maximum and tenth maximum
    respectively. Essentially this is the same wing clipping
    procedure, but the location of the clip is determined by the peak
    intensity instead of standard deviation.

    Note, in both cases, users are responsible for ensuring methods
    are called on reasonable data (in FW* cases we should be probably
    be using intensities). No corrections or pedestals are
    incorporated at this time, but could easily be added in the
    future.
    """

    _width_methods = {"FWHM": 0.5, "FWTM": 0.1}

    def __init__(self, rejection_sigma=3):
        """
        Instantiates with presribed `rejection_sigma`.

        :param rejection_sigma: Values falling outside
            `rejection_sigma` standard deviations are
            rejected. Defaults to 3.  Also accepts 'FWHM' and 'FWTM'
            corresponding to per-pixel full width at half maximum and
            tenth maximum respectively.

        """
        # Handle string `rejection_sigma`
        self._method = self._gaussian_method
        if isinstance(rejection_sigma, str):
            self.rejection_sigma = rejection_sigma.upper()
            self._method = self._width_method
            if self.rejection_sigma not in self._width_methods:
                raise ValueError(
                    f"`rejection_sigma` must be numeric or {self._width_methods.keys()}."
                )
        else:
            self.sigma = float(rejection_sigma)

    def _call(self, stack):
        """
        Dispatches rejection method based on `rejection_sigma`.
        """
        return self._method(stack)

    def _gaussian_method(self, stack):
        """
        Gaussian rejection.
        """

        # Compute the mean and standard deviations, pixel-wise.
        means = stack.mean(axis=0)
        std_devs = stack.std(axis=0)

        # Find values that lie outside per-pixel sigma deviations.
        outliers = np.abs(stack - means) > self.sigma * std_devs

        # Mask off the outliers.
        masked_stack = ma.masked_array(stack, mask=outliers)

        # Return mean without the outliers
        return masked_stack.mean(axis=0).data

    def _width_method(self, stack):
        """
        Width rejection.
        """

        # Compute per-pixel max.
        maxima = stack.max(axis=0)

        # Find the per-pixel value for clipping
        y = maxima * self._width_methods[self.rejection_sigma]

        # Compute outliers; values below clipping value.
        outliers = stack < y

        # Mask off the outliers.
        masked_stack = ma.masked_array(stack, mask=outliers)

        # Return mean withou the outliers
        return masked_stack.mean(axis=0).data


class WinsorizedImageStacker(ImageStacker):
    """
    Stack using Winsorization.

    Winsorizing is similar to SigmaRejectionImageStacker, except it
    admits a `percentile` and replaces rejected values with values at
    +/- `percentile`.
    """

    def __init__(self, percentile=0.1):
        """
        Instantiates WinsorizedImageStacker instance with prescribed
        `percentile`.

        See scipy docs for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html

        :param percentile: Float or tuple of floats.  Values must be
            [0,1] and represent the percentile used for trimming.  A
            tuple allows different lower and upper percentiles
            respectively. Example (0.1, 0.2) would Winsorize the lower
            10% and upper 20% of values.  In the case of a single
            value, the same value will be used for both upper and
            lower percentiles.
        """
        # Convert scalars to tuples.
        if not isinstance(percentile, tuple):
            percentile = (float(percentile),) * 2
        if (min(percentile) < 0) or (max(percentile) > 1):
            raise ValueError(f"`percentile` must be [0,1], passed {percentile}.")
        self.percentile = percentile

    def _call(self, stack):
        """
        Apply Winsorizing process the return the stack mean.
        """

        # Note, we intentionally disable `inplace` to avoid mutating
        # stack, just-in-case.
        stack = winsorize(stack, limits=self.percentile, inplace=False)

        # Return mean of Winsorized data.
        return stack.mean(axis=0).data


# The following will be blocked by ABC because they are not
# implemented yet, and are open to discussion/removal/implementation
# pending time.
class PoissonRejectionImageStacker(ImageStacker):
    """
    Stack using Poisson Rejection.
    """


class RobustChauvenetRejectionImageStacker(ImageStacker):
    """
    Stack using Robust Chauvenet Outlier Rejection.
    """
