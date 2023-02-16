import logging
from abc import ABC, abstractmethod
from heapq import heappush, heappushpop

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

    def _check_selection(self, selection, n_img):
        """
        Check that class `selection` is sane.

        :param selection: selection indices
        :param n_img: number of images available
        """
        # Check length
        if len(selection) != self._max_n:
            raise ValueError(
                f"Class selection must be len {self._max_n}, got {len(selection)}"
            )

        # Check indices [0, n_img)
        if np.max(selection) >= n_img or np.min(selection) < 0:
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
        return np.arange(self._max_n)


class RandomClassSelector(ClassSelector):
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
        return rng.choice(self._max_n, size=self._max_n, replace=False)


class ContrastClassSelector(ClassSelector):
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
        # Compute per class variance
        dist_var = np.var(distances[:, 1:], axis=1)

        # Compute the ordering, descending
        sorted_class_inds = np.argsort(dist_var)[::-1]

        # Return indices
        return sorted_class_inds


class GlobalClassSelector(ClassSelector):
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
        self._max_n = self.averager.src.n
        # We should hit every entry, but along the way identify missing values as nan.
        self.quality_scores = np.full(self._max_n, fill_value=float("nan"))

        # To be used for heapq (score, img_id, avg_im)
        # Note the default implementation is min heap, so `pop` returns smallest value.
        self.heap = []

        self._heap_limit_bytes = heap_size_limit_bytes
        self._heap_item_size = self.averager.src.L**2 + self.averager.dtype.itemsize

    def _heap_size(self):
        n = len(self.heap)
        if n == 0:
            return 0

        return n * self._heap_item_size

    def _select(self, classes, reflections, distances):
        for i in classes[:, 0]:
            im = self.averager.average(classes[i], self.reflections[i])

            quality_score = self._quality_function(im)

            # Assign in global quality score array
            self.quality_scores[i] = quality_score

            # Create cachable payload item
            item = (quality_score, i, im)

            if self._heap_item_size + self._heap_size < self._heap_limit_bytes:
                heappush(self.heap, item)
            else:
                # We're out of space,
                # pop and throw away the worst entry
                # after updating (pushing to) the heap.
                _ = heappushpop(self.heap, item)


class PriorRejection:
    def __init__(self, aggresive=True):
        self.excluded = set()
        self.aggresive = bool(aggresive)

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


class ContrastWithPriorClassSelector(PriorRejection, ContrastClassSelector):
    """
    Selects top classes based on highest contrast with prior rejection.
    """


class GlobalWithPriorClassSelector(PriorRejection, ClassSelector):
    """
    Extends ClassSelector for methods that require
    passing over all class average images and also PriorRejection.
    """


# Weight functions, todo, maybe make a class to hide the grid detail.
def banded_snr(
    img, grid=None, center_radius=0.5, outer_band_start=0.8, outer_band_end=1
):
    if isinstance(img, Image):
        img = img.asnumpy()[0]

    L = img.shape[-1]

    if grid is None:
        grid = grid_2d(L)

    center_mask = grid["r"] < center_radius
    outer_mask = (grid["r"] > outer_band_start) & (grid["r"] < outer_band_end)

    return np.var(img[center_mask]) / np.var(img[outer_mask])


def weighted_snr(img, grid=None, weight_function=None):
    """
    Take a 1d radial weight function.
    The function is expected to be defined [0,sqrt(2)].
    Function range is not limited, but should probably be [0,1].
    """

    if isinstance(img, Image):
        img = img.asnumpy()[0]

    if weight_function is None:

        def weight_function(r):
            # linear ramp
            return np.max(1 - r, 0)

        # def weight_function(r):
        #     # bump function (normalized to [0,1]
        #     if r >= 1:
        #         return 0
        #     else:
        #         return np.exp(-1/(1-r**2) + 1)

    L = img.shape[-1]

    if grid is None:
        grid = grid_2d(L)

    weights = np.frompyfunc(weight_function)(grid["r"])

    return img * weights
