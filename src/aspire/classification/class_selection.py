import logging
from abc import ABC, abstractmethod
from heapq import heappush, heappushpop
from operator import eq, le

import numpy as np

from aspire.classification import Averager2D
from aspire.image import Image
from aspire.utils import grid_2d

logger = logging.getLogger(__name__)

"""
Selecting the "best" classes is an area of active research.

Here we provide an abstract base class with two naive approaches as concrete implementations.

`RandomClassSelector` will select random indices from across the entire dataset,
with RNG controlled by `seed`.

`TopClassSelector' will select the first `n_classes` in order.
This may be useful for debugging and development.

Interested researchers are encouraged to implement their own selector
which will be evaluated at runtime immediately before `averages`
are computed by RIRClass2D.
"""


class ClassSelector(ABC):
    """
    Abstract interface for class selection.
    """

    @abstractmethod
    def _select(self, classes, reflections, distances):
        """
        Using the provided arguments, returns an array representing
        an index into `n` selected `classes`.

        This is the method developers should implement for a custom algorithm.

        :param n: number of classes to select
        :param classes: (n_img, n_nbor) array of image indices
        :param reflections: (n_img, n_nbor) boolean array of reflections between `classes[i][0]` and classes[i][j]`
        :param distances: (n_img, n_nbor) array of distances between `classes[i][0]` and classes[i][j]`

        :return: array of indices into `classes`
        """

    @classmethod
    @property
    @abstractmethod
    def quality_scores(cls):
        """
        All `ClassSelector` should assign a quality score
        array the same length as the selection output.

        For subclasses like TopClassSelector and RandomClassSelector
        where no quality information is derived, this array should be zeros.
        """

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

        # Compute max class id found in the network for sanity checking. when unknown.
        # Some subclasses may have a more explicit `_max_n` from an initializing `Source`.
        if not hasattr(self, "_max_n"):
            self._max_n = np.max(classes[:, 0])

        # Call the selection logic
        selection = self._select(classes, reflections, distances)

        # n_img should not exceed the largest index in first column of `classes`
        n_img = np.max(classes[:, 0]) + 1
        # Check values in selection are in bounds.
        self._check_selection(selection, n_img)

        return selection

    def _check_selection(self, selection, n_img, len_operator=eq):
        """
        Check that class `selection` is sane.

        :param selection: selection indices
        :param n_img: number of images available
        """
        # Check length, +1 for zero indexing
        if not len_operator(len(selection), self._max_n + 1):
            raise ValueError(
                f"Class selection must be {str(len_operator)} {self._max_n+1}, got {len(selection)}"
            )

        # Check indices [0, n_img)
        if np.max(selection) >= n_img or np.min(selection) < 0:
            raise ValueError(
                f"Class selection out of bounds [0, {n_img - 1}]"
                f"with [{np.min(selection)}, {np.max(selection)}]"
            )


class ClassSelectorUnranked(ClassSelector):
    @property
    def quality_scores(self):
        return np.zeros(self._max_n + 1)


class ClassSelectorRanked(ClassSelector):
    @property
    def quality_scores(self):
        if not hasattr(self, "_quality_scores"):
            raise RuntimeError("Must run `.select(...)` to compute quality_scores")
        return self._quality_scores


class TopClassSelector(ClassSelectorUnranked):
    def _select(self, classes, reflections, distances):
        """
        Returns classes in `Source` order.

        Mainly useful for debugging.
        """
        return np.arange(self._max_n + 1)  # +1 for zero indexing


class RandomClassSelector(ClassSelectorUnranked):
    def __init__(self, seed=None):
        """
        :param seed: RNG seed, de
        """
        self.seed = seed

    def _select(self, classes, reflections, distances):
        """
        Select random `n` classes from the population.
        """
        # Instantiate a random Generator
        rng = np.random.default_rng(self.seed)
        # Generate and return indices for random sample
        # +1 for zero indexing
        return rng.choice(self._max_n + 1, size=self._max_n + 1, replace=False)


class ContrastClassSelector(ClassSelectorRanked):
    """
    Selects top classes based on highest contrast,
    as estimated by variances of `distances`.

    Note that `distances` is the Nearest Neighbors distances,
    and in the case of RIR this is a small rotationally invariant feature
    vector.  For methods based on class average images, see subclasses of `GlobalClassSelector`.
    """

    def _select(self, classes, reflections, distances):
        # Compute per class variance
        dist_var = np.var(distances[:, 1:], axis=1)

        # Compute the ordering, descending
        sorted_class_inds = np.argsort(dist_var)[::-1]
        # Store the sorted quality scores (maps to selection output).
        self._quality_scores = dist_var[sorted_class_inds]

        # Return indices
        return sorted_class_inds


class DistanceClassSelector(ClassSelectorRanked):
    """
    Selects top classes based on lowest mean distance
    as estimated by `distances`.

    Note that `distances` is the Nearest Neighbors distances,
    and in the case of RIR this is a small rotationally invariant feature
    vector.  For methods based on class average images, see subclasses of `GlobalClassSelector`.
    """

    def _select(self, classes, reflections, distances):
        # Compute per class variance
        dist_mean = np.mean(distances[:, 1:], axis=1)

        # Compute the ordering, descending
        sorted_class_inds = np.argsort(dist_mean)[::-1]
        self._quality_scores = dist_mean[sorted_class_inds]

        # Return indices
        return sorted_class_inds


class GlobalClassSelector(ClassSelectorRanked):
    """
    Extends ClassSelector for methods that require
    passing over all class average images.
    """

    def __init__(self, averager, quality_function, heap_size_limit_bytes=2e9):
        """
        :param averager: An Averager2D subclass.
        :param quality_function: Function that takes an image and returns numeric quality score.
            This score will be used to sort the classes.
            For example, this module provides methods for Contrast and SNR.
        """
        self.averager = averager
        if not isinstance(self.averager, Averager2D):
            raise ValueError(
                f"`averager` should be instance of `Averger2D`, found {self.averager_src}."
            )

        self._quality_function = quality_function
        self._max_n = self.averager.src.n - 1
        # We should hit every entry, but along the way identify missing values as nan.
        self._quality_scores = np.full(self._max_n + 1, fill_value=float("nan"))

        # To be used for heapq (score, img_id, avg_im)
        # Note the default implementation is min heap, so `pop` returns smallest value.
        self.heap = []

        self._heap_limit_bytes = heap_size_limit_bytes
        self._heap_item_size = self.averager.src.L**2 + self.averager.dtype.itemsize

    @property
    def _heap_size(self):
        n = len(self.heap)
        if n == 0:
            return 0

        return n * self._heap_item_size

    @property
    def _heap_ids(self):
        """
        Return the image ids currently in the heap.
        """
        return [item[1] for item in self.heap]  # heap item=(score, img_id, img)

    @property
    def _heap_id_dict(self):
        """
        Return map of image ids to heap position currently in the heap.
        """
        return {
            item[1]: i for i, item in enumerate(self.heap)
        }  # heap item=(score, img_id, img)

    def _select(self, classes, reflections, distances):
        for i in classes[:, 0]:
            im = self.averager.average(classes[i], reflections[i])

            quality_score = self._quality_function(im)

            # Assign in global quality score array
            self._quality_scores[i] = quality_score

            # Create cachable payload item
            item = (quality_score, i, im)

            if self._heap_item_size + self._heap_size < self._heap_limit_bytes:
                heappush(self.heap, item)
            else:
                # We're out of space,
                # pop and throw away the worst entry
                # after updating (pushing to) the heap.
                _ = heappushpop(self.heap, item)

        # Now that we have computed the global quality_scores,
        # the selection ordering can be applied
        sorted_class_inds = np.argsort(self._quality_scores)[::-1]
        self._quality_scores = self._quality_scores[sorted_class_inds]
        return sorted_class_inds


class ClassRepulsion:
    """
    Mixin to overload class selection based on excluding
    classes we've alreay seen as neighbors of another class.

    If the classes are well sorted (by some measure of quality),
    we can assume the best representation is the first seen.

    :param aggressive: Aggresive mode will additionally exclude
        any new class containing a neighbor that has already
        been incorporated. Defaults to True.

    """

    def __init__(self, *args, aggressive=True, **kwargs):
        """
        Instatiates and sets `aggresive`. All other args and **kwagrs are pass through to super().

        :param aggressive: Aggresive mode will additionally exclude
            any new class containing a neighbor that has already
            been incorporated. Defaults to True.
        """

        self.excluded = set()
        self.aggressive = aggressive
        super().__init__(*args, **kwargs)

    def _select(self, classes, reflections, distances):
        # Get the indices sorted by the next resolving `_select` method.
        sorted_inds = super()._select(classes, reflections, distances)

        results = []
        for i in sorted_inds:
            # Skip when this class's base image has been seen in a prior class.
            if i in self.excluded:
                continue

            # Get the images in this class
            cls = classes[i]

            # Aggresive mode skips any class containing neighbors
            # we have have seen in prior class.
            if self.aggressive and self.excluded.intersection(cls[1:]):
                continue

            # Add images from this class to the exclusion list
            self.excluded.update(cls)

            results.append(i)

        return np.array(results, dtype=int)

    def _check_selection(self, selection, n_img):
        """
        Check that class `selection` is sane.

        Repulsion can reduce the number of classes (dramatically).

        :param selection: selection indices
        :param n_img: number of images available
        """
        return super()._check_selection(selection, n_img, len_operator=le)


class ContrastWithRepulsionClassSelector(ClassRepulsion, ContrastClassSelector):
    """
    Selects top classes based on highest contrast with prior rejection.
    """


class GlobalWithRepulsionClassSelector(ClassRepulsion, GlobalClassSelector):
    """
    Extends ClassSelector for methods that require
    passing over all class average images and also ClassRepulsion.
    """


class ImageQualityFunction(ABC):
    """
    A callable image quality scoring function.
    """

    _grid_cache = {}

    @abstractmethod
    def _function(self, img):
        """
        User defined 1d radial weight function.
        The function is expected to be defined for [0,sqrt(2)].
        Function range is not limited, but [0,1] is favorable.

        Develops can use the self._grid_cache for access to
        a grid_2d instance matching resolution of img.

        :param img: 2d numpy array
        :returns: Image quality score
        """

    def __call__(self, img):
        """
        Given an image instance or a 2D np array,
        builds a grid once if needed then
        calls the instance's weight function.

        :param img: `Image`, 2d numpy array,
            or 3d array where slow axis is stack axis.
            When called with an image stack>1, an array is returned.
        :returns: Image quality score(s)
        """

        if isinstance(img, Image):
            img = img.asnumpy()

        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        stack_len = img.shape[0]
        L = img.shape[-1]

        # Generate grid on first call so user can expect it exists.
        self._grid_cache.setdefault(L, grid_2d(L, dtype=img.dtype))

        # Call the function over img stack
        res = np.fromiter(map(self._function, img), dtype=img.dtype)

        if stack_len == 1:
            res = res[0]

        return res


class BandedSNRImageQualityFunction(ImageQualityFunction):
    def _function(self, img, center_radius=0.5, outer_band_start=0.8, outer_band_end=1):
        # Look up the precached grid.
        grid = self._grid_cache[img.shape[-1]]

        center_mask = grid["r"] < center_radius
        outer_mask = (grid["r"] > outer_band_start) & (grid["r"] < outer_band_end)

        return np.var(img[center_mask]) / np.var(img[outer_mask])


class WeightedImageQualityFunction(ImageQualityFunction):
    """
    A callable image quality scoring function using a radial grid weighted function.
    """

    def __init__(self):
        # Each weight function will need a seperate cache,
        # but the same function should be deterministic up to resolution.
        self._weights_cache = {}

    @abstractmethod
    def _weight_function(self, r):
        """
        Take a 1d radial weight function.
        The function is expected to be defined [0,sqrt(2)].
        Function range is not limited, but should probably be [0,1].
        """

    def weights(self, L):
        """
        Lookup weights for a given resolution L,
        computes and updates cache on first request.

        :param L: resolution pixels
        :returns: 2d weight array for LxL grid.
        """
        grid = self._grid_cache[L]
        return self._weights_cache.setdefault(
            L, np.frompyfunc(self.weight_function)(grid["r"])
        )

    def _function(self, img):
        L = img.shape[-1]

        # if we ignore the issue of dependence,
        # we could also do sum(per_pixel_variance * weights)
        return (img * img) * self.weights(L)


class RampImageQualityFunction(WeightedImageQualityFunction):
    def _weight_function(r):
        # linear ramp
        return np.max(1 - r, 0)


class BumpImageQualityFunction(WeightedImageQualityFunction):
    def _weight_function(r):
        # bump function (normalized to [0,1]
        if r >= 1:
            return 0
        else:
            return np.exp(-1 / (1 - r**2) + 1)
