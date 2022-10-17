import logging
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import ray
from ray.util.multiprocessing import Pool
from tqdm import tqdm, trange

from aspire import config
from aspire.classification.reddy_chatterji import reddy_chatterji_register
from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.utils.coor_trans import grid_2d
from aspire.utils.multiprocessing import num_procs_suggestion

logger = logging.getLogger(__name__)


class Averager2D(ABC):
    """
    Base class for 2D Image Averaging methods.
    """

    def __init__(self, composite_basis, src, num_procs=1, dtype=None):
        """
        :param composite_basis:  Basis to be used during class average composition (eg FFB2D)
        :param src: Source of original images.
        :param num_procs: Number of processes to use.
            `None` will attempt computing a suggestion based on machine resources.
            Note some underlying code may already use threading.
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.composite_basis = composite_basis
        self.src = src
        if dtype is None:
            if self.composite_basis:
                self.dtype = self.composite_basis.dtype
            elif self.src:
                self.dtype = self.src.dtype
            else:
                raise RuntimeError("You must supply a basis/src/dtype.")
        else:
            self.dtype = np.dtype(dtype)

        if num_procs is None:
            num_procs = num_procs_suggestion()
            # Only enable multiprocessing when several cores available
            if num_procs < 3:
                num_procs = 1
        elif not (isinstance(num_procs, int) and num_procs > 0):
            raise ValueError(
                f"num_procs should be a positive integer, passed {num_procs}."
            )
        self.num_procs = num_procs

        if self.src and self.dtype != self.src.dtype:
            logger.warning(
                f"{self.__class__.__name__} dtype {dtype}"
                f"does not match dtype of source {self.src.dtype}."
            )
        if self.composite_basis and self.dtype != self.composite_basis.dtype:
            logger.warning(
                f"{self.__class__.__name__} dtype {dtype}"
                f"does not match dtype of basis {self.composite_basis.dtype}."
            )

    @abstractmethod
    def average(
        self,
        classes,
        reflections,
        coefs=None,
    ):
        """
        Combines images using stacking in `self.composite_basis`.

        Subclasses should implement this.
        (Example EM algos use radically different averaging).

        Should return an Image source of synthetic class averages.

        :param classes: class indices, refering to src. (n_classes, n_nbor).
        :param reflections: Bool representing whether to reflect image in `classes`.
            (n_clases, n_nbor)
        :param coefs: Optional basis coefs (could avoid recomputing).
            (n_classes, coef_count)
        :return: Stack of synthetic class average images as Image instance.
        """

    def _cls_images(self, cls, src=None):
        """
        Util to return images as an array for class k (provided as array `cls` ),
        preserving the class/nbor order.

        :param cls: An iterable (0/1-D array or list) that holds the indices of images to align.
            In class averaging, this would be a class.
        :param src: Optionally override the src, for example, if you want to use a different
            source for a certain operation (ie alignment).
        """
        src = src or self.src

        n_nbor = cls.shape[-1]  # Includes zero'th neighbor

        images = np.empty((n_nbor, src.L, src.L), dtype=self.dtype)

        for i, index in enumerate(cls):
            images[i] = src.images[index].asnumpy()

        return images


class AligningAverager2D(Averager2D):
    """
    Subclass supporting averagers which perfom an aligning stage.
    """

    def __init__(
        self, composite_basis, src, alignment_basis=None, num_procs=1, dtype=None
    ):
        """
        :param composite_basis:  Basis to be used during class average composition (eg hi res Cartesian/FFB2D).
        :param src: Source of original images.
        :param alignment_basis: Optional, basis to be used only during alignment (eg FSPCA).
        :param num_procs: Number of processes to use.
            Note some underlying code may already use threading.
        :param dtype: Numpy dtype to be used during alignment.
        """

        super().__init__(
            composite_basis=composite_basis,
            src=src,
            num_procs=num_procs,
            dtype=dtype,
        )
        # If alignment_basis is None, use composite_basis
        self.alignment_basis = alignment_basis or self.composite_basis

        if not hasattr(self.composite_basis, "rotate"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s composite_basis {self.composite_basis} must provide a `rotate` method."
            )
        if not hasattr(self.composite_basis, "shift"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s composite_basis {self.composite_basis} must provide a `shift` method."
            )

    @abstractmethod
    def align(self, classes, reflections, basis_coefficients):
        """
        During this process `rotations`, `reflections`, `shifts` and
        `correlations` properties will be computed for aligners.

        `rotations` is an (n_classes, n_nbor) array of angles,
        which should represent the rotations needed to align images within
        that class. `rotations` is measured in radians.

        `shifts` is None or an (n_classes, n_nbor) array of 2D shifts
        which should represent the translation needed to best align the images
        within that class.

        `correlations` is an (n_classes, n_nbor) array representing
        a correlation like measure between classified images and their base
        image (image index 0).

        Subclasses of should implement and extend this method.

        :param classes: (n_classes, n_nbor) integer array of img indices.
        :param reflections: (n_classes, n_nbor) bool array of corresponding reflections,
        :param basis_coefficients: (n_img, self.alignment_basis.count) basis coefficients,

        :returns: (rotations, shifts, correlations)
        """

    def average(
        self,
        classes,
        reflections,
        coefs=None,
    ):
        """
        This subclass assumes we get alignment details from `align` method. Otherwise. see Averager2D.average
        """

        self.rotations, self.shifts, self.correlations = self.align(
            classes, reflections, coefs
        )

        n_classes, n_nbor = classes.shape

        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        def _innerloop(i):
            # Get coefs in Composite_Basis if not provided as an argument.
            if coefs is None:
                # Retrieve relavent images directly from source.
                neighbors_imgs = Image(self._cls_images(classes[i]))

                # Do shifts
                if self.shifts is not None:
                    neighbors_imgs.shift(self.shifts[i])

                neighbors_coefs = self.composite_basis.evaluate_t(neighbors_imgs)
            else:
                # Get the neighbors
                neighbors_ids = classes[i]
                neighbors_coefs = coefs[neighbors_ids]
                if self.shifts is not None:
                    neighbors_coefs = self.composite_basis.shift(
                        neighbors_coefs, self.shifts[i]
                    )

            # Rotate in composite_basis
            neighbors_coefs = self.composite_basis.rotate(
                neighbors_coefs, self.rotations[i], reflections[i]
            )

            # Averaging in composite_basis
            return np.mean(neighbors_coefs, axis=0)

        if self.num_procs <= 1:
            for i in tqdm(range(n_classes)):
                b_avgs[i] = _innerloop(i)
        else:
            logger.info(f"Starting Pool({self.num_procs})")
            ray.init(_temp_dir=config.ray.temp_dir)
            with Pool(self.num_procs) as p:
                results = p.map(_innerloop, range(n_classes))
            ray.shutdown()

            logger.info(f"Terminated Pool({self.num_procs}), unpacking results.")
            for i, result in enumerate(results):
                b_avgs[i] = result

        # Now we convert the averaged images from Basis to Cartesian.
        return ArrayImageSource(self.composite_basis.evaluate(b_avgs))


class BFRAverager2D(AligningAverager2D):
    """
    This perfoms a Brute Force Rotational alignment.

    For each class,
        constructs n_angles rotations of all class members,
        and then identifies angle yielding largest correlation(dot).
    """

    def __init__(
        self,
        composite_basis,
        src,
        alignment_basis=None,
        n_angles=360,
        num_procs=1,
        dtype=None,
    ):
        """
        See AligningAverager2D, adds:

        :param n_angles: Number of brute force rotations to attempt, defaults 360.
        """
        super().__init__(
            composite_basis, src, alignment_basis, num_procs=num_procs, dtype=dtype
        )

        self.n_angles = n_angles

        if not hasattr(self.alignment_basis, "rotate"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s alignment_basis {self.alignment_basis} must provide a `rotate` method."
            )

    def align(self, classes, reflections, basis_coefficients):
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
            if basis_coefficients is None:
                # Retrieve relavent images
                neighbors_imgs = Image(self._cls_images(classes[k]))
                # Evaluate_T into basis
                nbr_coef = self.composite_basis.evaluate_t(neighbors_imgs)
            else:
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

        return rotations, None, correlations


class BFSRAverager2D(BFRAverager2D):
    """
    This perfoms a Brute Force Shift and Rotational alignment.
    It is potentially expensive to brute force this search space.

    For each pair of x_shifts and y_shifts,
       Perform BFR

    Return the rotation and shift yielding the best results.
    """

    def __init__(
        self,
        composite_basis,
        src,
        alignment_basis=None,
        n_angles=360,
        n_x_shifts=1,
        n_y_shifts=1,
        num_procs=1,
        dtype=None,
    ):
        """
        See AligningAverager2D and BFRAverager2D, adds: `n_x_shifts`, `n_y_shifts`.

        Note that `n_x_shifts` and `n_y_shifts` are the number of shifts
        to perform in each direction.

        Example: n_x_shifts=1, n_y_shifts=0 would test {-1,0,1} X {0}.

        n_x_shifts=n_y_shifts=0 is the same as calling BFRAverager2D.

        :params n_angles: Number of brute force rotations to attempt, defaults 360.
        :params n_x_shifts: +- Number of brute force xshifts to attempt, defaults 1.
        :params n_y_shifts: +- Number of brute force xshifts to attempt, defaults 1.
        """
        super().__init__(
            composite_basis,
            src,
            alignment_basis,
            n_angles,
            num_procs=num_procs,
            dtype=dtype,
        )

        self.n_x_shifts = n_x_shifts
        self.n_y_shifts = n_y_shifts

        # Each shift will require calling the parent BFRAverager2D.align
        self._bfr_align = super().align

        if not hasattr(self.alignment_basis, "shift"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s alignment_basis {self.alignment_basis} must provide a `shift` method."
            )

    def align(self, classes, reflections, basis_coefficients):
        """
        See `AligningAverager2D.align`
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

        if basis_coefficients is None:
            # Retrieve image coefficients, this is bad, it load all images.
            # TODO: Refactor this s.t. the following code blocks and super().align
            #   only require coefficients relating to their class.  See _cls_images.
            basis_coefficients = self.composite_basis.evaluate_t(self.src.images[:])

        # We want to maintain the original coefs for the base images,
        #  because we will mutate them with shifts in the loop.
        original_coef = basis_coefficients[classes[:, 0], :]
        assert original_coef.shape == (n_classes, self.alignment_basis.count)

        # Loop over shift search space, updating best result
        for x, y in product(x_shifts, y_shifts):
            shift = np.array([x, y], dtype=int)
            logger.debug(f"Computing rotational alignment after shift ({x},{y}).")

            # Shift the coef representing the first (base) entry in each class
            #   by the negation of the shift
            # Shifting one image is more efficient than shifting every neighbor
            basis_coefficients[classes[:, 0], :] = self.alignment_basis.shift(
                original_coef, -shift
            )

            _rotations, _, _correlations = self._bfr_align(
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
                logger.debug("Initial rotational alignment complete (shift (0,0))")
                assert np.sum(improved_indices) == np.size(
                    classes
                ), f"{np.sum(improved_indices)} =?= {np.size(classes)}"
            else:
                logger.debug(
                    f"Shift ({x},{y}) complete. Improved {np.sum(improved_indices)} alignments."
                )

        return rotations, shifts, correlations


class ReddyChatterjiAverager2D(AligningAverager2D):
    """
    Attempts rotational estimation using Reddy Chatterji log polar Fourier cross correlation.
    Then attempts shift (translational) estimation using cross correlation.

    When averaging, performs rotations then shifts.

    Note, it may be possible to iterate this algorithm...

    Adopted from Reddy Chatterji  (1996)
    An FFT-Based Technique for Translation,
    Rotation, and Scale-Invariant Image Registration
    IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 5, NO. 8, AUGUST 1996

    This method intentionally does not use any of ASPIRE's basis
    so that it may be used as a reference for more ASPIRE approaches.
    """

    def __init__(
        self,
        composite_basis,
        src,
        alignment_src=None,
        num_procs=None,
        dtype=None,
    ):
        """
        :param composite_basis:  Basis to be used during class average composition.
        :param src: Source of original images.
        :param alignment_src: Optional, source to be used during class average alignment.
            Must be the same resolution as `src`.
        :param num_procs: Number of processes to use.
            `None` will attempt computing a suggestion based on machine resources.
            Note some underlying code may already use threading.
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.alignment_src = alignment_src or src

        # TODO, for accomodating different resolutions we minimally need to adapt shifting.
        # Outside of scope right now, but would make a nice PR later.
        if self.alignment_src.L != src.L:
            raise RuntimeError("Currently `alignment_src.L` must equal `src.L`")
        if self.alignment_src.dtype != src.dtype:
            raise RuntimeError("Currently `alignment_src.dtype` must equal `src.dtype`")

        self.mask = grid_2d(src.L, normalized=False)["r"] < src.L // 2

        super().__init__(
            composite_basis, src, composite_basis, num_procs=num_procs, dtype=dtype
        )

    def align(self, classes, reflections, basis_coefficients):
        """
        Performs the actual rotational alignment estimation,
        returning parameters needed for averaging.
        """

        # Admit simple case of single case alignment
        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        n_classes = classes.shape[0]

        # Instantiate matrices for results
        rotations = np.zeros(classes.shape, dtype=self.dtype)
        correlations = np.zeros(classes.shape, dtype=self.dtype)
        shifts = np.zeros((*classes.shape, 2), dtype=int)

        def _innerloop(k):
            # Get the array of images for this class, using the `alignment_src`.
            images_k = self._cls_images(classes[k], src=self.alignment_src)
            return reddy_chatterji_register(
                images_k,
                reflections[k],
                mask=self.mask,
                do_cross_corr_translations=True,
                dtype=self.dtype,
            )

        if self.num_procs <= 1:
            for k in trange(n_classes):
                rotations[k], shifts[k], correlations[k] = _innerloop(k)

        else:
            logger.info(f"Starting Pool({self.num_procs})")
            ray.init(_temp_dir=config.ray.temp_dir)
            with Pool(self.num_procs) as p:
                results = p.map(_innerloop, range(n_classes))
            ray.shutdown()

            logger.info(f"Terminated Pool({self.num_procs}), unpacking results.")
            for k, result in enumerate(results):
                rotations[k], shifts[k], correlations[k] = result

        return rotations, shifts, correlations

    def average(
        self,
        classes,
        reflections,
        coefs=None,
    ):
        """
        This averages classes performing rotations then shifts.
        Otherwise is similar to `AligningAverager2D.average`.
        """

        self.rotations, self.shifts, self.correlations = self.align(
            classes, reflections, coefs
        )

        n_classes, n_nbor = classes.shape

        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        def _innerloop(i):

            # Get coefs in Composite_Basis if not provided as an argument.
            if coefs is None:
                # Retrieve relavent images directly from source.
                neighbors_imgs = Image(self._cls_images(classes[i]))
                neighbors_coefs = self.composite_basis.evaluate_t(neighbors_imgs)
            else:
                # Get the neighbors
                neighbors_ids = classes[i]
                neighbors_coefs = coefs[neighbors_ids]

            # Rotate in composite_basis
            neighbors_coefs = self.composite_basis.rotate(
                neighbors_coefs, self.rotations[i], reflections[i]
            )

            # Note shifts are after rotation for this approach!
            if self.shifts is not None:
                neighbors_coefs = self.composite_basis.shift(
                    neighbors_coefs, self.shifts[i]
                )

            # Averaging in composite_basis
            return np.mean(neighbors_coefs, axis=0)

        for i in tqdm(range(n_classes)):
            b_avgs[i] = _innerloop(i)
        else:
            logger.info(f"Starting Pool({self.num_procs})")
            ray.init(_temp_dir=config.ray.temp_dir)
            with Pool(self.num_procs) as p:
                results = p.map(_innerloop, range(n_classes))
            ray.shutdown()

            logger.info(f"Terminated Pool({self.num_procs}), unpacking results.")
            for i, result in enumerate(results):
                b_avgs[i] = result

        # Now we convert the averaged images from Basis to Cartesian.
        return ArrayImageSource(self.composite_basis.evaluate(b_avgs))


class BFSReddyChatterjiAverager2D(ReddyChatterjiAverager2D):
    """
    Brute Force Shifts (Translations) - ReddyChatterji (Log-Polar) Rotations

    For each shift within `radius`, attempts rotational match using ReddyChatterji.
    When averaging, performs shift before rotations,

    Adopted from Reddy Chatterji  (1996)
    An FFT-Based Technique for Translation,
    Rotation, and Scale-Invariant Image Registration
    IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 5, NO. 8, AUGUST 1996

    This method intentionally does not use any of ASPIRE's basis
    so that it may be used as a reference for more ASPIRE approaches.
    """

    def __init__(
        self,
        composite_basis,
        src,
        alignment_src=None,
        radius=None,
        num_procs=None,
        dtype=None,
    ):
        """
        :param alignment_basis: Basis to be used during alignment.
            For current implementation of ReddyChatterjiAverager2D this should be `None`.
            Instead see `alignment_src`.
        :param src: Source of original images.
        :param composite_basis:  Basis to be used during class average composition.
        :param alignment_src: Optional, source to be used during class average alignment.
            Must be the same resolution as `src`.
        :param radius: Brute force translation search radius.
            Defaults to src.L//8.
        :param dtype: Numpy dtype to be used during alignment.

        :param num_procs: Number of processes to use.
            `None` will attempt computing a suggestion based on machine resources.
            Note some underlying code may already use threading.
        :param dtype: Numpy dtype to be used during alignment.
        """

        super().__init__(
            composite_basis,
            src,
            alignment_src,
            num_procs=num_procs,
            dtype=dtype,
        )

        # Assign search radius
        self.radius = radius or src.L // 8

    def align(self, classes, reflections, basis_coefficients):
        """
        Performs the actual rotational alignment estimation,
        returning parameters needed for averaging.
        """

        # Admit simple case of single case alignment
        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        n_classes, n_nbor = classes.shape
        L = self.alignment_src.L

        # Instantiate matrices for inner loop, and best results.
        rotations = np.zeros(classes.shape, dtype=self.dtype)
        correlations = np.ones(classes.shape, dtype=self.dtype) * -np.inf
        shifts = np.zeros((*classes.shape, 2), dtype=int)

        # We'll brute force all shifts in a grid.
        g = grid_2d(L, normalized=False)
        disc = g["r"] <= self.radius
        X, Y = g["x"][disc], g["y"][disc]

        def _innerloop(k):
            unshifted_images = self._cls_images(classes[k])
            # Instantiate matrices for inner loop, and best results.
            _rotations = np.zeros(classes.shape[1:], dtype=self.dtype)
            _correlations = np.ones(classes.shape[1:], dtype=self.dtype) * -np.inf
            _shifts = np.zeros((*classes.shape[1:], 2), dtype=int)

            for xs, ys in zip(X, Y):
                s = np.array([xs, ys])
                # Get the array of images for this class

                # Note we mutate `images` here with shifting,
                #   then later reddy_chatterji_register
                images = unshifted_images.copy()
                # Don't shift the base image
                images[1:] = Image(unshifted_images[1:]).shift(s).asnumpy()

                # returned shifts ignored since we are forcing shift of `s` above
                __rotations, _, __correlations = reddy_chatterji_register(
                    images,
                    reflections[k],
                    mask=self.mask,
                    do_cross_corr_translations=False,  # When forcing s, we skip cross corr translations
                    dtype=self.dtype,
                )

                # Where corr has improved
                #  update our rolling best results with this loop.
                improved = __correlations > _correlations
                _correlations = np.where(improved, __correlations, _correlations)
                _rotations = np.where(improved, __rotations, _rotations)
                _shifts = np.where(improved[..., np.newaxis], s, _shifts)
                logger.debug(f"Shift {s} has improved {np.sum(improved)} results")

            return _rotations, _shifts, _correlations

        if self.num_procs <= 1:
            for k in trange(n_classes):
                rotations[k], shifts[k], correlations[k] = _innerloop(k)
        else:
            logger.info(f"Starting Pool({self.num_procs})")
            ray.init(_temp_dir=config.ray.temp_dir)
            with Pool(self.num_procs) as p:
                results = p.map(_innerloop, range(n_classes))
            ray.shutdown()

            logger.info(f"Terminated Pool({self.num_procs}), unpacking results.")
            for k, result in enumerate(results):
                rotations[k], shifts[k], correlations[k] = result

        return rotations, shifts, correlations

    def average(
        self,
        classes,
        reflections,
        coefs=None,
    ):
        """
        See Averager2D.average.
        """
        # ReddyChatterjiAverager2D does rotations then shifts.
        # For brute force, we'd like shifts then rotations,
        #   as is done in general in AligningAverager2D
        return AligningAverager2D.average(self, classes, reflections, coefs)


class EMAverager2D(Averager2D):
    """
    Citation needed.
    """


class FTKAverager2D(Averager2D):
    """
    Factorization of the translation kernel for fast rigid image alignment.
    Rangan, A.V., Spivak, M., Anden, J., & Barnett, A.H. (2019).
    """
