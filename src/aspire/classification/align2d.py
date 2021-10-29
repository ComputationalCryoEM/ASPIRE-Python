import logging
from itertools import product

import numpy as np
from tqdm import trange

logger = logging.getLogger(__name__)


class Align2D:
    """
    Base class for 2D Image Alignment methods.
    """

    def __init__(self, basis, dtype):
        """
        :param basis: Basis to be used for any methods during alignment.
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.basis = basis
        if dtype is None:
            self.dtype = self.basis.dtype
        else:
            self.dtype = np.dtype(dtype)
            if self.dtype != self.basis.dtype:
                logger.warning(
                    f"Align2D basis.dtype {self.basis.dtype} does not match self.dtype {self.dtype}."
                )

    def align(self, classes, reflections, basis_coefficients):
        """
        Any align2D alignment method should take in the following arguments
        and return the described tuple.

        Generally, the returned `classes` and `reflections` should be same as
        the input.  They are passed through for convience,
        considering they would all be required for image output.

        Returned `rotations` is an (n_classes, n_nbor) array of angles,
        which should represent the rotations needed to align images within
        that class. `rotations` is measured in Radians.

        Returned `correlations` is an (n_classes, n_nbor) array representing
        a correlation like measure between classified images and their base
        image (image index 0).

        Returned `shifts` is None or an (n_classes, n_nbor) array of 2D shifts
        which should represent the translation needed to best align the images
        within that class.

        Subclasses of `align` should extend this method with optional arguments.

        :param classes: (n_classes, n_nbor) integer array of img indices
        :param refl: (n_classes, n_nbor) bool array of corresponding reflections
        :param coef: (n_img, self.pca_basis.count) compressed basis coefficients

        :returns: (classes, reflections, rotations, shifts, correlations)
        """
        raise NotImplementedError("Subclasses must implement align.")


class BFRAlign2D(Align2D):
    """
    This perfoms a Brute Force Rotational alignment.

    For each class,
        constructs n_angles rotations of all class members,
        and then identifies angle yielding largest correlation(dot).
    """

    def __init__(self, basis, n_angles=359, dtype=None):
        """
        :params basis: Basis providing a `rotate` method.
        :params n_angles: Number of brute force rotations to attempt, defaults 359.
        """
        super().__init__(basis, dtype)

        self.n_angles = n_angles

        if not hasattr(self.basis, "rotate"):
            raise RuntimeError(
                f"BFRAlign2D's basis {self.basis} must provide a `rotate` method."
            )

    def align(self, classes, reflections, basis_coefficients):
        """
        See `Align2D.align`
        """
        # Admit simple case of single case alignment
        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        n_classes, n_nbor = classes.shape

        # Construct array of angles to brute force.
        test_angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)

        # Instantiate matrices for results
        rotations = np.empty(classes.shape, dtype=self.dtype)
        correlations = np.empty(classes.shape, dtype=self.dtype)
        results = np.empty((n_nbor, self.n_angles))

        for k in trange(n_classes):

            # Get the coefs for these neighbors
            nbr_coef = basis_coefficients[classes[k]]

            for i, angle in enumerate(test_angles):
                # Rotate the set of neighbors by angle,
                rotated_nbrs = self.basis.rotate(nbr_coef, angle, reflections[k])

                # then store dot between class base image (0) and each nbor
                for j, nbor in enumerate(rotated_nbrs):
                    results[j, i] = np.dot(nbr_coef[0], nbor)

            # Now along each class, find the index of the angle reporting highest correlation
            angle_idx = np.argmax(results, axis=1)

            # Store that angle as our rotation for this image
            rotations[k, :] = test_angles[angle_idx]

            # Also store the correlations for each neighbor
            for j in range(n_nbor):
                correlations[k, j] = results[j, angle_idx[j]]

        # None is placeholder for shifts
        return classes, reflections, rotations, None, correlations


class BFSRAlign2D(BFRAlign2D):
    """
    This perfoms a Brute Force Shift and Rotational alignment.
    It is potentially expensive to brute force this search space.

    For each pair of x_shifts and y_shifts,
       Perform BFR

    Return the rotation and shift yielding the best results.
    """

    def __init__(self, basis, n_angles=359, n_x_shifts=1, n_y_shifts=1, dtype=None):
        """
        Note that n_x_shifts and n_y_shifts are the number of shifts to perform
        in each direction.

        Example: n_x_shifts=1, n_y_shifts=0 would test {-1,0,1} X {0}.

        n_x_shifts=n_y_shifts=0 is the same as calling BFRAlign2D.

        :params basis: Basis providing a `shift` and `rotate` method.
        :params n_angles: Number of brute force rotations to attempt, defaults 359.
        :params n_x_shifts: +- Number of brute force xshifts to attempt, defaults 1.
        :params n_y_shifts: +- Number of brute force xshifts to attempt, defaults 1.
        """
        super().__init__(basis, n_angles, dtype)

        self.n_x_shifts = n_x_shifts
        self.n_y_shifts = n_y_shifts

        if not hasattr(self.basis, "shift"):
            raise RuntimeError(
                f"BFSRAlign2D's basis {self.basis} must provide a `shift` method."
            )

        # Each shift will require calling the parent BFRAlign2D.align
        self._bfr_align = super().align

    def align(self, classes, reflections, basis_coefficients):
        """
        See `Align2D.align`
        """

        # Admit simple case of single case alignment
        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        n_classes = classes.shape[0]

        # Compute the shifts. Roll array so 0 is first.
        x_shifts = np.roll(
            np.arange(-self.n_x_shifts, self.n_x_shifts + 1), -self.n_x_shifts
        )
        y_shifts = np.roll(
            np.arange(-self.n_y_shifts, self.n_y_shifts + 1), -self.n_y_shifts
        )
        # Above rolls should force initial pair of shifts to (0,0).
        # This is done primarily in case of a tie later we would take unshifted.
        assert (x_shifts[0], y_shifts[0]) == (0, 0)

        # These arrays will incrementally store our best alignment.
        rotations = np.empty(classes.shape, dtype=self.dtype)
        correlations = np.ones(classes.shape, dtype=self.dtype) * -np.inf
        shifts = np.empty((*classes.shape, 2), dtype=int)

        # We want to maintain the original coefs for the base images,
        #  because we will mutate them with shifts in the loop.
        original_coef = basis_coefficients[classes[:, 0], :]
        assert original_coef.shape == (n_classes, self.basis.count)

        # Loop over shift search space, updating best result
        for x, y in product(x_shifts, y_shifts):
            shift = np.array([x, y], dtype=int)
            logger.info(f"Computing Rotational alignment after shift ({x},{y}).")

            # Shift the coef representing the first (base) entry in each class
            #   by the negation of the shift
            # Shifting one image is more efficient than shifting every neighbor
            basis_coefficients[classes[:, 0], :] = self.basis.shift(
                original_coef, -shift
            )

            _, _, _rotations, _, _correlations = self._bfr_align(
                classes, reflections, basis_coefficients
            )

            # Each class-neighbor pair may have a best shift-rot from a different shift.
            # Test and update
            improved_indices = _correlations > correlations
            rotations[improved_indices] = _rotations[improved_indices]
            correlations[improved_indices] = _correlations[improved_indices]
            shifts[improved_indices] = shift

            # Restore unshifted base coefs
            basis_coefficients[classes[:, 0], :] = original_coef

            if (x, y) == (0, 0):
                logger.info("Initial rotational alignment complete (shift (0,0))")
                assert np.sum(improved_indices) == np.size(
                    classes
                ), f"{np.sum(improved_indices)} =?= {np.size(classes)}"
            else:
                logger.info(
                    f"Shift ({x},{y}) complete. Improved {np.sum(improved_indices)} alignments."
                )

        return classes, reflections, rotations, shifts, correlations


class EMAlign2D(Align2D):
    """
    Citation needed.
    """

    def __init__(self, basis, dtype=None):
        super().__init__(basis, dtype)


class FTKAlign2D(Align2D):
    """
    Factorization of the translation kernel for fast rigid image alignment.
    Rangan, A.V., Spivak, M., Anden, J., & Barnett, A.H. (2019).
    """

    def __init__(self, basis, dtype=None):
        super().__init__(basis, dtype)

    def align(self, classes, reflections, basis_coefficients):
        raise NotImplementedError
