import logging
from itertools import product

import numpy as np
from tqdm import trange

logger = logging.getLogger(__name__)


class Class2D:
    """
    Base class for 2D Image Classification methods.
    """

    def __init__(
        self,
        src,
        n_nbor=100,
        n_classes=50,
        alignment_implementation="bfr",
        alignment_opts=None,
        seed=None,
        dtype=None,
    ):
        """
        Base constructor of an object for classifying 2D images.

        :param n_nbor: Number of nearest neighbors to compute.
        :param n_classes: Number of class averages to return.
        :param alignment_implementation: See `alignment`.
        :param alignment_opts: Optional implementation specific configuration options. See `alignment`.
        :param seed: Optional RNG seed to be passed to random methods, (example Random NN).


        """
        self.src = src

        if dtype is not None:
            self.dtype = np.dtype(dtype)
            if self.dtype != self.src.dtype:
                logger.warning(
                    f"Class2D src.dtype {self.src.dtype} does not match self.dtype {self.dtype}."
                )
        else:
            self.dtype = self.src.dtype

        self.n_nbor = n_nbor
        self.n_classes = n_classes
        self.alignment_implementation = alignment_implementation
        self.alignment_opts = alignment_opts
        self.seed = seed

        alignment_implementations = {
            "bfr": self._bfr_align,
            "bfsr": self._bfsr_align,
        }

        if alignment_implementation not in alignment_implementations:
            raise ValueError(
                f"Provided alignment_implementation={alignment_implementation}"
                f" not in {alignment_implementations.keys()}."
            )
        self._alignment = alignment_implementations[alignment_implementation]

    def alignment(self, classes, refl, coef, alignment_opts=None):
        """
        Any class averagiing alignment method should take in the following arguments and return the tuple described.

        The returned `classes` and `refl` should be same as the input.

        Returned `rot` is an (n_classes, n_nbor) array of angles which should represent the rotations needed to align images within that class. `rot` is measure in Radians.

        Returned `corr` is an (n_classes, n_nbor) array that should represent a correlation like measure between classified images and their base image (image index 0).

        Returned `shifts` is None or an (n_classes, n_nbor) array of 2D shifts which should represent the translation needed to best align the images within that class.


        Alignment implementations may admit specific conifguration options using an optional `alignment_opts` dictionary.

        :param classes: (n_classes, n_nbor) integer array of indices
        :param refl: (n_classes, n_nbor) bool array of reflections
        :param coef: (n_img, self.pca_basis.count) array of compressed basis coefficients.

        :returns: (classes, refl, rot, corr, shifts)
        """

        # _alignment is assigned during initialization.
        return self._alignment(classes, refl, coef, alignment_opts)

    def _bfsr_align(self, classes, refl, coef, alignment_opts=None):
        """
        This perfoms a Brute Force Shift and Rotational alignment.

        For each pair of x_shifts and y_shifts,
           Perform BFR

        Return the rotation and shift yielding the best results.
        """

        # Unpack any configuration options, or get defaults.
        if alignment_opts is None:
            alignment_opts = {}
        # Default shift search space of +- 1 in X and Y
        n_x_shifts = alignment_opts.get("n_x_shifts", 1)
        n_y_shifts = alignment_opts.get("n_y_shifts", 1)

        # Compute the shifts. Roll array so 0 is first.
        x_shifts = np.roll(np.arange(-n_x_shifts, n_x_shifts + 1), -n_x_shifts)
        y_shifts = np.roll(np.arange(-n_y_shifts, n_y_shifts + 1), -n_y_shifts)
        assert (x_shifts[0], y_shifts[0]) == (0, 0)

        # These arrays will incrementally store our best alignment.
        rots = np.empty(classes.shape, dtype=self.dtype)
        corr = np.ones(classes.shape, dtype=self.dtype) * -np.inf
        shifts = np.empty((*classes.shape, 2), dtype=int)

        # We want to maintain the original coefs for the base images,
        #  because we will mutate them with shifts in the loop.
        original_coef = coef[classes[:, 0], :]
        assert original_coef.shape == (self.n_classes, self.pca_basis.count)

        # Loop over shift search space, updating best result
        for x, y in product(x_shifts, y_shifts):
            shift = np.array([x, y], dtype=int)
            logger.info(f"Computing Rotational alignment after shift ({x},{y}).")

            # Shift the coef representing the first (base) entry in each class
            #   by the negation of the shift
            # Shifting one image is more efficient than shifting every neighbor
            coef[classes[:, 0], :] = self.pca_basis.shift(original_coef, -shift)

            _, _, _rots, _, _corr = self._bfr_align(classes, refl, coef, alignment_opts)

            # Each class-neighbor pair may have a best shift-rot from a different shift.
            # Test and update
            improved_indices = _corr > corr
            rots[improved_indices] = _rots[improved_indices]
            corr[improved_indices] = _corr[improved_indices]
            shifts[improved_indices] = shift

            # Restore unshifted base coefs
            coef[classes[:, 0], :] = original_coef

            if (x, y) == (0, 0):
                logger.info("Initial rotational alignment complete (shift (0,0))")
                assert np.sum(improved_indices) == np.size(
                    classes
                ), f"{np.sum(improved_indices)} =?= {np.size(classes)}"
            else:
                logger.info(
                    f"Shift ({x},{y}) complete. Improved {np.sum(improved_indices)} alignments."
                )

        return classes, refl, rots, shifts, corr

    def _bfr_align(self, classes, refl, coef, alignment_opts=None):
        """
        This perfoms a Brute Force Rotational alignment.

        For each class,
            constructs n_angles rotations of all class members,
            and then identifies angle yielding largest correlation(dot).

        For params, see `align`.

        Configurable `alignment_opts`:
        `n_angles` sets the number of brute force rotations to attempt.
        Defaults `n_angles=359`.
        """

        # Configure any alignment options, otherwise use defaults.
        if alignment_opts is None:
            alignment_opts = {}
        n_angles = alignment_opts.get("n_angles", 359)

        # Construct array of angles to brute force.
        test_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # Instantiate matrices for results
        rots = np.empty(classes.shape, dtype=self.dtype)
        corr = np.empty(classes.shape, dtype=self.dtype)
        results = np.empty((self.n_nbor, n_angles))

        for k in trange(self.n_classes):

            # Get the coefs for these neighbors
            nbr_coef = coef[classes[k]]

            for i, angle in enumerate(test_angles):
                # Rotate the set of neighbors by angle,
                rotated_nbrs = self.pca_basis.rotate(nbr_coef, angle, refl[k])

                # then store dot between class base image (0) and each nbor
                for j, nbor in enumerate(rotated_nbrs):
                    results[j, i] = np.dot(nbr_coef[0], nbor)

            # Now along each class, find the index of the angle reporting highest correlation
            angle_idx = np.argmax(results, axis=1)

            # Store that angle as our rotation for this image
            rots[k, :] = test_angles[angle_idx]

            # Also store the correlations for each neighbor
            for j in range(self.n_nbor):
                corr[k, j] = results[j, angle_idx[j]]

        # None is placeholder for shifts
        return classes, refl, rots, None, corr
