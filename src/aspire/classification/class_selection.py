"""
Selecting the "best" classes is an area of active research.

Here we provide an abstract base class with two naive approaches as
concrete implementations.

`RandomClassSelector` will select random indices from across the
entire dataset, with RNG controlled by `seed`.

`TopClassSelector' will select the first `n_classes` in order.  This
may be useful for debugging and development.

Additionally we provide a few methods that have been used
historically, along with a few classes which should aid in
constructing new methods.
"""

import logging
from abc import ABC, abstractmethod
from heapq import heappush, heappushpop
from operator import eq, le

import numpy as np

from aspire.classification import Averager2D
from aspire.image import Image
from aspire.utils import grid_2d

logger = logging.getLogger(__name__)


class ClassSelector(ABC):
    """
    Abstract interface for class selection.
    """

    @abstractmethod
    def _select(self, classes, reflections, distances):
        """
        Using the provided arguments, returns an array representing
        an index into selected `classes`.

        This is the method developers should implement for a custom algorithm.

        :param classes: (n_img, n_nbor) array of image indices
        :param reflections: (n_img, n_nbor) boolean array of reflections between `classes[i][0]` and classes[i][j]`
        :param distances: (n_img, n_nbor) array of distances between `classes[i][0]` and classes[i][j]`

        :return: array of indices into `classes`
        """

    @property
    def quality_scores(self):
        """
        All `ClassSelector` should assign a quality score array the
        same length as the selection output.

        Function range is currently not limited, but [0,1] is
        favorable.  Currently there is not an expectation that one
        quality scoring system relates to another, or that the score
        is a proper metric. Quality scores are only required to be a
        self consistent ordering.

        For subclasses like TopClassSelector and RandomClassSelector
        where no quality information is derived, the associated
        `_quality_scores` should be set to zeros by `_select`.
        """
        # Remind the user to run selection first.  We could run it
        # automatically, but some of the class selectors may take a
        # long time.
        if not hasattr(self, "_quality_scores"):
            raise RuntimeError("Must run `.select(...)` to compute quality_scores")
        return self._quality_scores

    def select(self, classes, reflections, distances):
        """
        Using the provided arguments, calls internal `_select`
        method, checks selection is sane, and returns an array representing
        an ordered index into `classes`.

        :param classes: (n_img, n_nbor) array of image indices
        :param reflections: (n_img, n_nbor) boolean array of reflections between `classes[i][0]` and classes[i][j]`
        :param distances: (n_img, n_nbor) array of distances between `classes[i][0]` and classes[i][j]`

        :return: array of indices into `classes`
        """

        # Compute max class id found in the network for sanity
        # checking. Base class defaults to the `n` known from the
        # initializing `Source`.  This `n` may be reduced through
        # class selection.
        if not hasattr(self, "n"):
            self.n = np.max(classes[:, 0]) + 1  # +1 for zero indexing

        # Call the selection logic
        self.selection = self._select(classes, reflections, distances)

        # n_img should not exceed the largest index in first column of `classes`.
        n_img = np.max(classes[:, 0]) + 1  # +1 for zero indexing
        # Check values in selection are in the bounds.
        self._check_selection(self.selection, n_img)

        return self.selection

    def _check_selection(self, selection, n_img, len_operator=eq):
        """
        Check that class `selection` is sane.

        :param selection: selection indices
        :param n_img: number of images available
        :param len_operator: Operation used for comparison.
            Defaults to equality (`eq`).
        """
        # Check length, +1 for zero indexing
        if not len_operator(len(selection), self.n):
            raise ValueError(
                f"Class selection must be {str(len_operator)} {self.n}, got {len(selection)}"
            )

        # Check indices [0, n_img)
        if (np.max(selection) >= n_img) or (np.min(selection) < 0):
            raise ValueError(
                f"Class selection out of bounds [0, {n_img - 1}]"
                f"with [{np.min(selection)}, {np.max(selection)}]"
            )


class TopClassSelector(ClassSelector):
    def _select(self, classes, reflections, distances):
        """
        Returns classes in `Source` order.

        Mainly useful for debugging.
        """
        # Assign uniform quality.
        self._quality_scores = np.zeros(self.n)
        # Return same indices as initial source.
        return np.arange(self.n)


class RandomClassSelector(ClassSelector):
    def __init__(self, seed=None):
        """
        :param seed: RNG seed, de
        """
        self.seed = seed

    def _select(self, classes, reflections, distances):
        """
        Select random classes from the population.
        """
        # Assign uniform quality.
        self._quality_scores = np.zeros(self.n)

        # Instantiate a random Generator
        rng = np.random.default_rng(self.seed)
        # Generate and return indices for random sample
        # +1 for zero indexing
        return rng.choice(self.n, size=self.n, replace=False)


class NeighborVarianceClassSelector(ClassSelector):
    """
    Selects classes based on variances of `distances`.

    Note that `distances` is the Nearest Neighbors distances,
    and in the case of RIR this is a small rotationally invariant feature
    vector.  For methods based on class average images, see subclasses of `GlobalClassSelector`.
    """

    def _select(self, classes, reflections, distances):
        # Compute per class variance
        dist_var = np.var(distances, axis=1)

        # Compute the ordering, ascending
        sorted_class_inds = np.argsort(dist_var)
        # Store the sorted quality scores (maps to selection output).
        self._quality_scores = dist_var[sorted_class_inds]

        # Return indices
        return sorted_class_inds


class DistanceClassSelector(ClassSelector):
    """
    Selects top classes based on lowest mean distance
    as estimated by `distances`.

    Note that `distances` is the Nearest Neighbors distances,
    and in the case of RIR this is a small rotationally invariant feature
    vector.  For methods based on class average images, see subclasses of `GlobalClassSelector`.
    """

    def _select(self, classes, reflections, distances):
        # Compute per class distance.
        # Skips self distance (0).
        dist_mean = np.mean(distances[:, 1:], axis=1)

        # Compute the ordering, ascending
        sorted_class_inds = np.argsort(dist_mean)
        self._quality_scores = dist_mean[sorted_class_inds]

        # Return indices
        return sorted_class_inds


class _HeapItem:
    """
    Organize our heap entries.
    """

    def __init__(self, value, index, image):
        """
        Initialize a heap item.

        :param value: Float value used for heap ranking
        :param index: Image index
        :param image: Image object
        """
        self.value = float(value)
        self.index = int(index)
        self.image = image

    def __lt__(self, other):
        """
        This comparator is used by `heapq` to order and maintain the heap.
        """
        return self.value < other.value

    @staticmethod
    def nbytes(img_size, dtype):
        """
        Computes a rough size of _HeapItem based on assigned
        attributes. Note doesn't include python object overhead.

        :param img_size: Image size in pixels, (img_size, img_size).
        :param dtype: Image datatype.

        :return: Sum of attribute sizes.
        """
        # Allow passing dtype directly, or instance of dtype.
        dtype = np.dtype(dtype)
        return img_size**2 * dtype.itemsize + 16


class GlobalClassSelector(ClassSelector):
    """
    Extends ClassSelector for methods that require
    passing over all class average images.
    """

    def __init__(self, averager, quality_function, heap_size_limit_bytes=2e9):
        """
        Initializes a GlobalClassSelector.

        Because GlobalClassSelectors must compute all class averages,
        a heap cache maintains the top class averages as scored by
        `quality_function`.  If you have the memory, recommend setting
        the cache to be > n_classes*img_size*img_size*img.dtype.

        :param averager: An Averager2D subclass.
        :param quality_function: Function that takes an image and
            returns numeric quality score.  This score will be used to
            sort the classes.  User's may provide a callable function,
            but extending `ImageQualityFunction` is recommended.  For
            example, this module provides methods for variance and SNR
            based quality.
        :param heap_size_limit_bytes: Max heap size in Bytes.
            Defaults 2GB, 0 will disable.
        """
        self.averager = averager
        if not isinstance(self.averager, Averager2D):
            raise ValueError(
                f"`averager` should be instance of `Averager2D`, found {self.averager}."
            )

        self._quality_function = quality_function
        if not callable(self._quality_function):
            raise ValueError(
                "`quality_function` must be a callable function.  See ImageQualityFunction."
            )

        self.n = self.averager.src.n
        # We should eventually compute all `n` entries,
        # but start with identifying missing values as nan.
        self._quality_scores = np.full(self.n, fill_value=float("nan"))

        # To be used for heapq (score, img_id, avg_im)
        # Note the default implementation is min heap, so `pop` returns smallest value.
        self.heap = []

        self._heap_limit_bytes = int(heap_size_limit_bytes)
        self._heap_item_size = self.averager.src.L**2 * self.averager.dtype.itemsize

    @property
    def _heap_size(self):
        """
        Return estimate of heap_size.  Note this is not the exact
        heap_size, as it doesn't include the overhead for python
        objects.
        """
        n = len(self.heap)
        item_size = _HeapItem.nbytes(
            img_size=self.averager.composite_basis.nres, dtype=self.averager.dtype
        )

        return n * item_size

    @property
    def heap_ids(self):
        """
        Return the image ids currently in the heap.
        """
        return [item.index for item in self.heap]

    @property
    def heap_idx_map(self):
        """
        Return map of image ids to heap position currently in the heap.
        """
        return {item.index: i for i, item in enumerate(self.heap)}

    def _select(self, classes, reflections, distances):
        for i, im in enumerate(self.averager.average(classes, reflections)):
            quality_score = self._quality_function(im)

            # Assign in global quality score array
            self._quality_scores[i] = quality_score

            # Skip heap if single item is larger than heap limit.
            # Implies `self._heap_limit_bytes = 0` disables heap.
            if self._heap_item_size > self._heap_limit_bytes:
                continue

            # Create cachable payload item
            item = _HeapItem(quality_score, i, im)

            if self._heap_item_size + self._heap_size < self._heap_limit_bytes:
                # There is space to push another entry onto heap
                heappush(self.heap, item)
            elif self._heap_size < self._heap_limit_bytes:
                # Heap is currently out of space,
                # pop and throw away the worst entry
                # after updating (pushing to) the heap.
                _ = heappushpop(self.heap, item)

        # Now that we have computed the global quality_scores,
        # the selection ordering can be applied, descending
        sorted_class_inds = np.argsort(self._quality_scores)[::-1]
        self._quality_scores = self._quality_scores[sorted_class_inds]
        return sorted_class_inds


# TODO: When a consistent measure of distance is implemented by
# preceeding components we can implement exclusion based on neighbor
# distances or other ideas (VDM?) as different ClassRepulsion like
# classes.
class GreedyClassRepulsionMixin:
    """
    Mixin to overload class selection based on excluding
    classes we've already seen as neighbors of another class.

    If the classes are well sorted (by some measure of quality),
    we assume the best representation is the first seen.
    """

    def __init__(self, *args, **kwargs):
        """
        Sets optional `exclude_k`. All other args and **kwargs are
        passed to super().

        GreedyClassRepulsionMixin is similar to `cryo_select_subset` from
        MATLAB, but MATLAB found `exclude_k` iteratively based on a
        desired result set size.

        :param exclude_k: Number of neighbors from each class to
            exclude.  Defaults to all neighbors.
        """
        # Pop of the parameter unique to GreedyClassRepulsionMixin.
        self.exclude_k = kwargs.pop("exclude_k", None)

        # Instantiate an empty set to hold our excluded indices.
        self.excluded = set()

        # Pass everything else through to the super __init__.
        super().__init__(*args, **kwargs)

    def _select(self, classes, reflections, distances):
        """
        Overload the ClassSelector._select
        """
        # Get the indices sorted by the super's `_select` method.
        sorted_inds = super()._select(classes, reflections, distances)

        # If exclude_k is not provided, default to exluding all
        # neighbors seen.
        k = self.exclude_k or classes.shape[-1]

        results = []
        for i in sorted_inds:
            # Skip when this class's base image has been seen in a
            # prior class.
            if i in self.excluded:
                continue

            # Get the images in this class `i`, and add class images
            # up to `k` the exclusion list.
            self.excluded.update(classes[i, :k])

            results.append(i)

        return np.array(results, dtype=int)

    def _check_selection(self, selection, n_img):
        """
        Check that class `selection` is sane.

        Repulsion can reduce the number of classes (dramatically),
        so we need to adjust the checking operation.

        :param selection: selection indices
        :param n_img: number of images available
        """
        return super()._check_selection(selection, n_img, len_operator=le)


class NeighborVarianceWithRepulsionClassSelector(
    GreedyClassRepulsionMixin, NeighborVarianceClassSelector
):
    """
    Selects top classes based on highest contrast with GreedyClassRepulsionMixin.
    """


class GlobalWithRepulsionClassSelector(GreedyClassRepulsionMixin, GlobalClassSelector):
    """
    Extends ClassSelector for methods that require
    passing over all class average images and also GreedyClassRepulsionMixin.
    """


class ImageQualityFunction(ABC):
    """
    A callable image quality scoring function.

    The main advantage to using this class is to gain access to a grid
    caching and Image/Numpy conversion.
    """

    _grid_cache = {}

    @abstractmethod
    def _function(self, img):
        """
        User defined scoring function.  Function domain and range is
        currently not limited, but [0,1] is favorable range.

        Developers can use the self._grid_cache for access to a
        grid_2d instance matching resolution of img.

        :param img: 2d Numpy array :returns: Image quality score
        """

    def __call__(self, img):
        """
        Given an image instance or a 2D np array,
        calls the instance's weight function.

        :param img: `Image`, 2d Numpy array,
            or 3d array where slow axis is stack axis.
            When called with an image stack>1, an array is returned.
        :returns: Image quality score(s)
        """

        if isinstance(img, Image):
            img = img.asnumpy()

        # Allow the function to be used on a single image presented as
        # a 2d Numpy array, as well as a stack.
        if img.ndim == 2:
            stack_len = 0  # singleton
            img = img[np.newaxis, :, :]
        else:
            stack_len = img.shape[0]

        # Image size.
        L = img.shape[-1]

        # Generate grid on first call as needed.
        self._grid_cache.setdefault(L, grid_2d(L, dtype=img.dtype))

        # Call the function over img stack.
        res = np.fromiter(map(self._function, img), dtype=img.dtype)

        # Return singleton when given a singleton (2d array).
        if stack_len == 0:
            res = res[0]

        return res


class VarianceImageQualityFunction(ImageQualityFunction):
    """
    Computes the variance of pixels.
    """

    def _function(self, img):
        """
        Scoring function based on variance.

        :param img: Input image as 2d Numpy array.

        :return: Pixel variance.
        """
        return np.var(img)


class BandedSNRImageQualityFunction(ImageQualityFunction):
    """
    Computes the ratio of variance of central pixels of image to
    pixels in a configurable outer band.
    """

    def _function(self, img, center_radius=0.5, outer_band_start=0.8, outer_band_end=1):
        """Configurable scoring function.

        :param img: Input image as 2d Numpy array.
        :param center_radius: Proportion of image radius defining the center.
        :param outer_band_start: Proportion of image radius defining
            the inner boundary of band.
        :param outer_band_end: Proportion of image radius defining the
            outer boundary of band.  Must be larger than outer_band_start.

        :return: Ratio central variance to outer band variance.
        """
        # Image Size
        L = img.shape[-1]

        # Look up the grid.
        grid = self._grid_cache[L]

        center_mask = grid["r"] < center_radius
        outer_mask = (grid["r"] > outer_band_start) & (grid["r"] < outer_band_end)
        if not np.any(outer_mask):
            # outer_mask is empty, need to correct outer_band_{start,end}
            raise RuntimeError(
                f"Band of ({outer_band_start}, {outer_band_end}) empty for image size {L}, adjust band boundaries."
            )

        return np.var(img[center_mask]) / np.var(img[outer_mask])


class BandpassImageQualityFunction(ImageQualityFunction):
    """
    Replicate behavior of MATLAB `cryo_sort_stack_bandpass` method.
    """

    # TODO, sort by polar Fourier bandpass of Image.
    # This was implemented in MATLAB, but appears unused.
    # Note, we may derive a similar method for non global use using bispectrum
    #   (if we do bookkeeping during compression).


class WeightedImageQualityMixin(ABC):
    """
    Extends ImageQualityFunction with a radial grid weighted function
    for use in user defined `_function` calls.
    """

    def __init__(self):
        # Each weight function will need a seperate cache.
        self._weights_cache = {}
        super().__init__()

    @abstractmethod
    def _weight_function(self, r):
        """
        Take a 1d radial weight function.
        The function is expected to be defined [0,sqrt(2)].
        Function range is not limited, but should probably be [0,1].

        :param r: Radius (grid).
        :return: 2D grid of weights
        """

    def weights(self, L):
        """
        Returns  2D array of weights for a given resolution L.
        Computes and caches on first request.

        :param L: resolution pixels

        :return: 2d weight array for LxL grid.
        """
        grid = self._grid_cache[L]
        # frompyfunc performs the broadcast.
        return self._weights_cache.setdefault(
            L, np.frompyfunc(self._weight_function, 1, 1)(grid["r"])
        )

    def _function(self, img):
        """
        Apply weights to `img` before calling underlying `_function`.

        Users may override this to use weights in more sophisticated
        manner.
        """
        img = img * self.weights(img.shape[-1])
        return super()._function(img)


# These classes are provided as helpers/examples.
class RampWeightedImageQualityMixin(WeightedImageQualityMixin):
    """
    ImageQualityMixin to apply a linear ramp.
    """

    def _weight_function(self, r):
        # linear ramp
        return np.max(1 - r, 0)


class BumpWeightedImageQualityMixin(WeightedImageQualityMixin):
    """
    ImageQualityMixin to apply a [0,1] bump function.
    """

    def _weight_function(self, r):
        # bump function (normalized to [0,1]).
        if r >= 1:
            return 0
        else:
            return np.exp(-1 / (1 - r**2) + 1)


class BumpWeightedVarianceImageQualityFunction(
    BumpWeightedImageQualityMixin, VarianceImageQualityFunction
):
    """
    Computes the variance of pixels after weighting with Bump function.
    """


class RampWeightedVarianceImageQualityFunction(
    RampWeightedImageQualityMixin, VarianceImageQualityFunction
):
    """
    Computes the variance of pixels after weighting with Ramp function.
    """
