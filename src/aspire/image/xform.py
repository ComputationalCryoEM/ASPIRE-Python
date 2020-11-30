import logging

import numpy as np
from joblib import Memory

from aspire.image import Image
from aspire.operators import PowerFilter, ZeroFilter
from aspire.utils.random import randn

logger = logging.getLogger(__name__)


class Xform:
    """
    An Xform is anything that implements a `forward` method (and an `adjoint` method, in the case of a LinearXform),
    that takes in a square Image object and spits out a square Image object corresponding to forward/adjoint operations.

    It does this by setting up whatever internal data structures it needs to set up in its constructor.
    Xform objects usually set up data structures that are typically *larger* than the depth of the Image
    object that they expect to encounter during any invocation of `forward` or `adjoint`.

    At runtime, it gets an Image object (a thin wrapper on a n x L x L ndarray),
    as well as numeric `indices` (a numpy array of index values) that correspond to index values (the 'window') of
    the incoming Image object within the context of all images (i.e. an Image object of N x L x L, where N represents
    the number of total images that can ever pass through this Xform.

    At runtime, The Xform object may choose to ignore `indices` altogether (e.g. a Xform that downsamples all incoming
    images to a constant resolution won't care what `indices` they correspond to), or may do something with it (e.g. a
    Xform that shifts images by varying offsets depending on their indices).
    """

    class XformActiveContextManager:
        """
        This inner class allows us to temporarily enable/disable a Xform object, by tweaking its `active`
        attribute on enter/exit.
        """

        def __init__(self, xform, active):
            self.xform = xform
            self.active = active

        def __enter__(self):
            self.xform_old_state = self.xform.active
            self.xform.active = self.active

        def __exit__(self, exc_type, exc_value, exc_traceback):
            self.xform.active = self.xform_old_state

    def __init__(self, active=True):
        """
        Create a Xform object that works at a specific resolution.
        :param active: A boolean indicating whether the Xform is active. True by default.
        """
        self.active = active

    def forward(self, im, indices=None):
        """
        Apply forward transformation for this Xform object to an Image object.
        :param im: The incoming Image object of depth `n`, on which to apply the forward transformation.
        :param indices: The indices to use within this Xform. If unspecified, [0..n) is used.
        :return: An Image object after applying the forward transformation.
        """
        if not self.active:
            return im
        if indices is None:
            indices = np.arange(im.n_images)
        return self._forward(im, indices=indices)

    def _forward(self, im, indices):
        raise NotImplementedError(
            "Subclasses must implement the _forward method applicable to im/indices."
        )

    def enabled(self):
        """
        Enable this Xform in a context manager, regardless of its `active` attribute value.
        :return: A context manager in which this Xform is enabled.
        """
        return Xform.XformActiveContextManager(self, active=True)

    def disabled(self):
        """
        Disable this Xform in a context manager, regardless of its `active` attribute value.
        :return: A context manager in which this Xform is disabled.
        """
        return Xform.XformActiveContextManager(self, active=False)

    def __str__(self):
        return self.__class__.__name__


class LinearXform(Xform):
    """
    A LinearXform is a Xform that additionally provides an `adjoint` method, similar to the `forward` method.
    """

    def adjoint(self, im, indices=None):
        """
        Apply adjoint transformation for this Xform object to an Image object.
        :param im: The incoming Image object of depth `n`, on which to apply the adjoint transformation.
        :param indices: The indices to use within this Xform. If unspecified, [0..n) is used.
        :return: An Image object after applying the adjoint transformation.
        """
        if not self.active:
            return im
        if indices is None:
            indices = np.arange(im.n_images)
        return self._adjoint(im, indices=indices)

    def _adjoint(self, im, indices):
        raise NotImplementedError(
            "Subclasses must implement the _adjoint method applicable to im/indices."
        )


class SymmetricXform(LinearXform):
    """
    A Symmetric Xform is a LinearXform where the forward/adjoint operations are identical.
    """

    def _adjoint(self, im, indices=None):
        return self._forward(im, indices)


class Multiply(SymmetricXform):
    """
    A Xform that changes the amplitudes of a stack of 2D images (in the form of an Image object) by multiplying all
    pixels of a single 2D  image by a constant factor.
    """

    def __init__(self, factor):
        """
        Initialize a Multiply Xform using specified factors
        :param factor: A float/int or an ndarray of scalar factors to use for amplitude multiplication.
        """
        super().__init__()
        self.multipliers = np.array(factor)

    def _forward(self, im, indices):
        if self.multipliers.size == 1:  # if we have a scalar multiplier
            im_new = im * self.multipliers
        else:
            im_new = im * self.multipliers[indices]

        return im_new

    def __str__(self):
        if self.multipliers.size == 1:
            return f"Multiply ({self.multipliers})"
        return "Multiply (multiple)"


class Shift(LinearXform):
    """
    A Xform that shifts pixels of a stack of 2D images (in the form of an Image object)by offsetting all pixels of a
    single 2D image by constant x/y offsets.
    """

    def __init__(self, shifts):
        """
        Initialize a Shift Xform using a Numpy array of shift values.
        :param shifts: An ndarray of shape (2) or (n, 2)
        """
        super().__init__()
        self.shifts = np.array(shifts)

    def _forward(self, im, indices):
        if self.shifts.ndim == 1:
            im_new = im.shift(self.shifts)
        else:
            im_new = im.shift(self.shifts[indices])

        return im_new

    def _adjoint(self, im, indices):
        if self.shifts.ndim == 1:
            im_new = im.shift(-self.shifts)
        else:
            im_new = im.shift(-self.shifts[indices])

        return im_new

    def __str__(self):
        if self.shifts.ndim == 1:
            return f"Shift ({self.shifts})"
        return "Shift (multiple)"


class Downsample(LinearXform):
    """
    A Xform that downsamples an Image object to a resolution specified by this Xform's resolution.
    """

    def __init__(self, resolution):
        self.resolution = resolution
        super().__init__()

    def _forward(self, im, indices):
        return im.downsample(self.resolution)

    def _adjoint(self, im, indices):
        # TODO: Implement up-sampling with zero-padding
        raise NotImplementedError("Adjoint of downsampling not implemented yet.")

    def __str__(self):
        return f"Downsample (Resolution {self.resolution})"


class FilterXform(SymmetricXform):
    """
    A `Xform` that applies a single `Filter` object to a stack of 2D images (as an Image object).
    """

    def __init__(self, filter):
        """
        Initialize the Filter `Xform` using a `Filter` object
        :param filter: An object of type `aspire.operators.Filter`
        """
        super().__init__()
        self.filter = filter

    def _forward(self, im, indices):
        return im.filter(self.filter)

    def __str__(self):
        return f"FilterXform ({self.filter})"


class Add(Xform):
    """
    A Xform that add the density of a stack of 2D images (in the form of an Image object)
    by an offset value.
    """

    def __init__(self, addend):
        """
        Initialize an Add Xform using a Numpy array of predefined values.
        :param addend: An float/int or an ndarray of shape (n,)
        """
        super().__init__()
        self.addend = np.array(addend)

    def _forward(self, im, indices):
        if self.addend.size == 1:
            im_new = im + self.addend
        else:
            im_new = im + self.addend[indices]

        return im_new

    def __str__(self):
        if self.addend.size == 1:
            return f"Add ({self.addend})"
        return "Add (multiple)"


class LambdaXform(Xform):
    """
    A `Xform` that applies a predefined function to a stack of 2D images (as an Image object).
    """

    def __init__(self, lambda_fun, *args, **kwargs):
        """
        Initialize the `LambdaXform` using a lambda function

        :param lambda_fun: Predefine lambda function
        :param *args: Positional arguments
        :param **kwargs: Keyword arguments
        """
        super().__init__()
        self.lambda_fun = lambda_fun
        self.args = args
        self.kwargs = kwargs

    def _forward(self, im, indices):
        im_in = im.asnumpy()
        im_out = self.lambda_fun(im_in, *self.args, **self.kwargs)

        return Image(im_out)

    def __str__(self):
        return f"LambdaXform ({self.lambda_fun.__name__})"


class NoiseAdder(Xform):
    """
    A Xform that adds white noise, optionally passed through a Filter object, to all incoming images.
    """

    def __init__(self, seed=0, noise_filter=None):
        """
        Initialize the random state of this NoiseAdder using specified values.
        :param seed: The random seed used to generate white noise
        :param noise_filter: An optional aspire.operators.Filter object to use to filter the generated white noise.
            By default, a ZeroFilter is used, generating no noise.
        """
        super().__init__()
        self.seed = seed
        noise_filter = noise_filter or ZeroFilter()
        self.noise_filter = PowerFilter(noise_filter, power=0.5)

    def _forward(self, im, indices):
        im = im.copy()

        for i, idx in enumerate(indices):
            # Note: The following random seed behavior is directly taken from MATLAB Cov3D code.
            random_seed = self.seed + 191 * (idx + 1)
            im_s = randn(2 * im.res, 2 * im.res, seed=random_seed)
            im_s = Image(im_s).filter(self.noise_filter)[0]
            im[i] += im_s[: im.res, : im.res]

        return im


class IndexedXform(Xform):
    """
    An IndexedXform is a Xform where individual Xform objects are used at specific indices of the incoming Image object.

    This extra layer of abstraction is used because individual Xform objects are typically capable of dealing with
    a stack of images (an Image object) in an efficient manner. The IndexedXform class groups the invocation of each
    of its unique Xform objects, so that calls to individual Xform objects within it is minimized, and equals the
    number of unique Xforms found.
    """

    def __init__(self, unique_xforms, indices=None):
        if indices is None:
            indices = np.arange(0, len(unique_xforms))
        else:
            # Ensure we're dealing with a numpy array since we'll be utilizing numpy indexing later on.
            indices = np.array(indices)
            assert np.min(indices) >= 0
            assert np.max(indices) < len(unique_xforms)

        super().__init__()

        self.n_indices = len(indices)
        self.indices = indices
        self.n_xforms = len(unique_xforms)
        self.unique_xforms = unique_xforms

        # A list of references to individual Xform objects, with possibly multiple references pointing to
        # the same Xform object.
        self.xforms = [unique_xforms[i] for i in indices]

    def _indexed_operation(self, im, indices, which):
        """
        Apply either a forward or adjoint transformations to `im`, depending on the value of the 'which' parameter.
        :param im: The incoming Image object on which to apply the forward or adjoint transformations.
        :param indices: The indices of the transformations to apply.
        :param which: The attribute indicating the function handle to obtain from underlying `Xform` objects.
            Typically either 'forward' or 'adjoint'.
        :return: An Image object as a result of applying forward or adjoint transformation to `im`.
        """
        # Ensure that we will be able to apply all transformers to the image
        assert (
            self.n_indices >= im.n_images
        ), f"Can process Image object of max depth {self.n_indices}. Got {im.n_images}."

        im_data = np.empty_like(im.asnumpy())

        # For each individual transformation
        for i, xform in enumerate(self.unique_xforms):
            # Get the indices corresponding to that transformation
            idx = np.flatnonzero(self.indices == i)
            # For the incoming Image object, find out which transformation indices are applicable
            idx = np.intersect1d(idx, indices)
            # For the transformation indices we found, find the indices in the Image object that we'll use
            im_data_indices = np.flatnonzero(np.isin(indices, idx))
            # Apply the transformation to the selected indices in the Image object
            if len(im_data_indices) > 0:
                fn_handle = getattr(xform, which)
                im_data[im_data_indices] = fn_handle(
                    Image(im[im_data_indices])
                ).asnumpy()

        return Image(im_data)

    def _forward(self, im, indices):
        return self._indexed_operation(im, indices, "forward")


class LinearIndexedXform(IndexedXform, LinearXform):
    def _adjoint(self, im, indices):
        return self._indexed_operation(im, indices, "adjoint")


def _apply_xform(xform, im, indices, adjoint=False):
    """
    A simple global function (i.e. not a method) that is capable of being cached by joblib.Memory object's `cache`
    method.
    """
    if not adjoint:
        logger.info("  Applying " + str(xform))
        return xform.forward(im, indices=indices)
    else:
        logger.info("  Applying Adjoint " + str(xform))
        return xform.adjoint(im, indices=indices)


class Pipeline(Xform):
    """
    A `Pipeline` is a `Xform` made up of individual transformation steps (i.e. multiple `Xform` objects).
    The `Pipeline`, just like any other `Xform`, can be run in the `forward` (or `adjoint` mode, in the case of a
    `LinearPipeline`.

    In addition to keeping client-side code clean, a major advantage of `Pipeline` is that individual steps of the
    pipeline can be cached transparently by the `Pipeline`, providing significant performance advantages for steps that
    are performed repeatedly (especially during development while setting up these pipelines) on any Image/Xform pair.
    This caching uses `joblib.Memory` object behind the scenes, but is disabled by default.
    """

    def __init__(self, xforms=None, memory=None):
        """
        Initialize a `Pipeline` with `Xform` objects.
        :param xforms: An iterable of Xform objects to use in the Pipeline.
        :param memory: None for no caching (default), or the location of a directory to use to cache steps of the
            pipeline.
        """
        self.xforms = xforms or []
        self.memory = memory
        self.active = True

    def __str__(self):
        return "Apply pipeline: " + " ".join([f"{xform}" for xform in self.xforms])

    def add_xform(self, xform):
        """
        Add a single `Xform` object at the end of the pipeline.
        :param xform: A `Xform` object.
        :return: None
        """
        self.xforms.append(xform)

    def add_xforms(self, xforms):
        """
        Add multiple `Xform` objects at the end of the pipeline.
        :param xform: An iterable of `Xform` objects.
        :return: None
        """
        self.xforms.extend(xforms)

    def reset(self):
        """
        Reset the pipeline by removing all `Xform`s.
        :return: None
        """
        self.xforms = []

    def _forward(self, im, indices):
        memory = Memory(location=self.memory, verbose=0)
        _apply_transform_cached = memory.cache(_apply_xform)

        logger.info("Applying forward transformations in pipeline")
        for xform in self.xforms:
            im = _apply_transform_cached(xform, im, indices, False)
        logger.info("All forward transformations applied")

        return im


class LinearPipeline(Pipeline, LinearXform):
    def _adjoint(self, im, indices):
        memory = Memory(location=self.memory, verbose=0)
        _apply_transform_cached = memory.cache(_apply_xform)

        logger.info("Applying adjoint transformations in pipeline")
        for xform in self.xforms[::-1]:
            im = _apply_transform_cached(xform, im, indices, True)
        logger.info("All adjoint transformations applied")

        return im
