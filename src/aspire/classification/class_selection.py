import logging
from abc import ABC, abstractmethod

import numpy as np

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
    def _select(self, n, classes, reflections, distances):
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

    def select(self, n, classes, reflections, distances):
        """
        Using the provided arguments, calls internal `_select`
        method, checks selection is sane, and returns an array representing
        an index into `n` selected `classes`.

        :param n: number of classes to select
        :param classes: (n_img, n_nbor) array of image indices
        :param reflections: (n_img, n_nbor) boolean array of reflections between `classes[i][0]` and classes[i][j]`
        :param distances: (n_img, n_nbor) array of distances between `classes[i][0]` and classes[i][j]`

        :return: array of indices into `classes`
        """

        # Call the selection logic
        selection = self._select(n, classes, reflections, distances)

        # n_img should not exceed the largest index in first column of `classes`
        n_img = np.max(classes[:, 0]) + 1
        # Check values in selection are in bounds.
        self._check_selection(selection, n, n_img)

        return selection

    def _check_selection(self, selection, n_classes, n_img):
        """
        Check that class `selection` is sane.

        :param selection: selection indices
        :param n_classes: number of classes expected
        :param n_img: number of images available
        """
        # Check length
        if len(selection) != n_classes:
            raise ValueError(
                f"Class selection must be len {n_classes}, got {len(selection)}"
            )

        # Check indices [0, n_img)
        if np.max(selection) >= n_img or np.min(selection) < 0:
            raise ValueError(
                f"Class selection out of bounds [0, {n_img - 1}]"
                f"with [{np.min(selection)}, {np.max(selection)}]"
            )


class TopClassSelector(ClassSelector):
    def _select(self, n, classes, reflections, distances):
        """
        Selects the first  `n` classes.

        This may be useful for debugging or when a `Source` is known
        to be sufficiently randomized.
        """
        return np.arange(n)


class RandomClassSelector(ClassSelector):
    def __init__(self, seed=None):
        """
        :param seed: RNG seed, de
        """
        self.seed = seed

    def _select(self, n, classes, reflections, distances):
        """
        Select random `n` classes from the population.
        """
        # Instantiate a random Generator
        rng = np.random.default_rng(self.seed)
        # Generate and return indices for random sample
        return rng.choice(np.max(classes[:, 0]), size=n, replace=False)
