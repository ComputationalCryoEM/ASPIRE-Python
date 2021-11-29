import logging
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
from tqdm import tqdm, trange

from aspire.image import Image
from aspire.source import ArrayImageSource

logger = logging.getLogger(__name__)


class Align2D(ABC):
    """
    Base class for 2D Image Alignment methods.
    """

    def __init__(self, alignment_basis, source, composite_basis=None, dtype=None):
        """
        :param alignment_basis: Basis to be used during alignment (eg FSPCA)
        :param source: Source of original images.
        :param composite_basis:  Basis to be used during class average composition (eg FFB2D)
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.alignment_basis = alignment_basis
        # if composite_basis is None, use alignment_basis
        self.composite_basis = composite_basis or self.alignment_basis
        self.src = source
        if dtype is None:
            self.dtype = self.alignment_basis.dtype
        else:
            self.dtype = np.dtype(dtype)
            if self.dtype != self.alignment_basis.dtype:
                logger.warning(
                    f"Align2D alignment_basis.dtype {self.alignment_basis.dtype} does not match self.dtype {self.dtype}."
                )

    @abstractmethod
    def align(self, classes, reflections, basis_coefficients):
        """
        Any align2D alignment method should take in the following arguments
        and return aligned images.

        During this process `rotations`, `reflections`, `shifts` and
        `correlations` propeties will be computed for aligners
        that implement them.

        `rotations` would be an (n_classes, n_nbor) array of angles,
        which should represent the rotations needed to align images within
        that class. `rotations` is measured in Radians.

        `correlations` is an (n_classes, n_nbor) array representing
        a correlation like measure between classified images and their base
        image (image index 0).

        `shifts` is None or an (n_classes, n_nbor) array of 2D shifts
        which should represent the translation needed to best align the images
        within that class.

        Subclasses of `align` should extend this method with optional arguments.

        :param classes: (n_classes, n_nbor) integer array of img indices
        :param refl: (n_classes, n_nbor) bool array of corresponding reflections
        :param coef: (n_img, self.pca_basis.count) compressed basis coefficients

        :returns: Image instance (stack of images)
        """


class AveragedAlign2D(Align2D):
    """
    Subclass supporting aligners which perform averaging during output.
    """

    def align(self, classes, reflections, basis_coefficients):
        """
        See Align2D.align
        """
        # Correlations are currently unused, but left for future extensions.
        cls, ref, rot, shf, corrs = self._align(
            classes, reflections, basis_coefficients
        )
        return self.average(cls, ref, rot, shf), cls, ref, rot, shf, corrs

    def average(
        self,
        classes,
        reflections,
        rotations,
        shifts=None,
        coefs=None,
    ):
        """
        Combines images using averaging in provided `basis`.

        :param classes: class indices (refering to src). (n_img, n_nbor)
        :param reflections: Bool representing whether to reflect image in `classes`
        :param rotations: Array of in-plane rotation angles (Radians) of image in `classes`
        :param shifts: Optional array of shifts for image in `classes`.
        :coefs: Optional Fourier bessel coefs (avoids recomputing).
        :return: Stack of Synthetic Class Average images as Image instance.
        """
        n_classes, n_nbor = classes.shape

        # TODO: don't load all the images here.
        imgs = self.src.images(0, self.src.n)
        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        for i in tqdm(range(n_classes)):
            # Get the neighbors
            neighbors_ids = classes[i]

            # Get coefs in Composite_Basis if not provided as an argument.
            if coefs is None:
                neighbors_imgs = Image(imgs[neighbors_ids])
                if shifts is not None:
                    neighbors_imgs.shift(shifts[i])
                neighbors_coefs = self.composite_basis.evaluate_t(neighbors_imgs)
            else:
                neighbors_coefs = coefs[neighbors_ids]
                if shifts is not None:
                    neighbors_coefs = self.composite_basis.shift(
                        neighbors_coefs, shifts[i]
                    )

            # Rotate in composite_basis
            neighbors_coefs = self.composite_basis.rotate(
                neighbors_coefs, rotations[i], reflections[i]
            )

            # Averaging in composite_basis
            b_avgs[i] = np.mean(neighbors_coefs, axis=0)

        # Now we convert the averaged images from Basis to Cartesian.
        return ArrayImageSource(self.composite_basis.evaluate(b_avgs))


class BFRAlign2D(AveragedAlign2D):
    """
    This perfoms a Brute Force Rotational alignment.

    For each class,
        constructs n_angles rotations of all class members,
        and then identifies angle yielding largest correlation(dot).
    """

    def __init__(
        self, alignment_basis, source, composite_basis=None, n_angles=359, dtype=None
    ):
        """
        :params alignment_basis: Basis providing a `rotate` method.
        :param source: Source of original images.
        :params n_angles: Number of brute force rotations to attempt, defaults 359.
        """
        super().__init__(alignment_basis, source, composite_basis, dtype)

        self.n_angles = n_angles

        if not hasattr(self.alignment_basis, "rotate"):
            raise RuntimeError(
                f"BFRAlign2D's alignment_basis {self.alignment_basis} must provide a `rotate` method."
            )

    def _align(self, classes, reflections, basis_coefficients):
        """
        Performs the actual rotational alignment estimation,
        returning parameters needed for averaging.
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
                rotated_nbrs = self.alignment_basis.rotate(
                    nbr_coef, angle, reflections[k]
                )

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

        return classes, reflections, rotations, None, correlations


class BFSRAlign2D(BFRAlign2D):
    """
    This perfoms a Brute Force Shift and Rotational alignment.
    It is potentially expensive to brute force this search space.

    For each pair of x_shifts and y_shifts,
       Perform BFR

    Return the rotation and shift yielding the best results.
    """

    def __init__(
        self,
        alignment_basis,
        source,
        composite_basis=None,
        n_angles=359,
        n_x_shifts=1,
        n_y_shifts=1,
        dtype=None,
    ):
        """
        Note that n_x_shifts and n_y_shifts are the number of shifts to perform
        in each direction.

        Example: n_x_shifts=1, n_y_shifts=0 would test {-1,0,1} X {0}.

        n_x_shifts=n_y_shifts=0 is the same as calling BFRAlign2D.

        :params alignment_basis: Basis providing a `shift` and `rotate` method.
        :params n_angles: Number of brute force rotations to attempt, defaults 359.
        :params n_x_shifts: +- Number of brute force xshifts to attempt, defaults 1.
        :params n_y_shifts: +- Number of brute force xshifts to attempt, defaults 1.
        """
        super().__init__(alignment_basis, source, composite_basis, n_angles, dtype)

        self.n_x_shifts = n_x_shifts
        self.n_y_shifts = n_y_shifts

        if not hasattr(self.alignment_basis, "shift"):
            raise RuntimeError(
                f"BFSRAlign2D's alignment_basis {self.alignment_basis} must provide a `shift` method."
            )

        # Each shift will require calling the parent BFRAlign2D._align
        self._bfr_align = super()._align

    def _align(self, classes, reflections, basis_coefficients):
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
        assert original_coef.shape == (n_classes, self.alignment_basis.count)

        # Loop over shift search space, updating best result
        for x, y in product(x_shifts, y_shifts):
            shift = np.array([x, y], dtype=int)
            logger.info(f"Computing Rotational alignment after shift ({x},{y}).")

            # Shift the coef representing the first (base) entry in each class
            #   by the negation of the shift
            # Shifting one image is more efficient than shifting every neighbor
            basis_coefficients[classes[:, 0], :] = self.alignment_basis.shift(
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


class FTKAlign2D(Align2D):
    """
    Factorization of the translation kernel for fast rigid image alignment.
    Rangan, A.V., Spivak, M., Anden, J., & Barnett, A.H. (2019).
    """
