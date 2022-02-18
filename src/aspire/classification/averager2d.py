import logging
from abc import ABC, abstractmethod
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import difference_of_gaussians, window
from skimage.transform import rotate, warp_polar
from tqdm import tqdm, trange

from aspire.image import Image
from aspire.numeric import fft
from aspire.source import ArrayImageSource
from aspire.utils.coor_trans import grid_2d

logger = logging.getLogger(__name__)


class Averager2D(ABC):
    """
    Base class for 2D Image Averaging methods.
    """

    def __init__(self, composite_basis, src, dtype=None):
        """
        :param composite_basis:  Basis to be used during class average composition (eg FFB2D)
        :param src: Source of original images.
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

        if self.src and self.dtype != self.src.dtype:
            logger.warning(
                f"{self.__class__.__name__} dtype {dtype}"
                "does not match dtype of source {self.src.dtype}."
            )
        if self.composite_basis and self.dtype != self.composite_basis.dtype:
            logger.warning(
                f"{self.__class__.__name__} dtype {dtype}"
                "does not match dtype of basis {self.composite_basis.dtype}."
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
            images[i] = src.images(index, 1).asnumpy()

        return images


class AligningAverager2D(Averager2D):
    """
    Subclass supporting averagers which perfom an aligning stage.
    """

    def __init__(self, composite_basis, src, alignment_basis=None, dtype=None):
        """
        :param composite_basis:  Basis to be used during class average composition (eg hi res Cartesian/FFB2D).
        :param src: Source of original images.
        :param alignment_basis: Optional, basis to be used only during alignment (eg FSPCA).
        :param dtype: Numpy dtype to be used during alignment.
        """

        super().__init__(
            composite_basis=composite_basis,
            src=src,
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

        rotations, shifts, _ = self.align(classes, reflections, coefs)

        n_classes, n_nbor = classes.shape

        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        for i in tqdm(range(n_classes)):

            # Get coefs in Composite_Basis if not provided as an argumen.
            if coefs is None:
                # Retrieve relavent images directly from source.
                neighbors_imgs = Image(self._cls_images(classes[i]))

                # Do shifts
                if shifts is not None:
                    neighbors_imgs.shift(shifts[i])

                neighbors_coefs = self.composite_basis.evaluate_t(neighbors_imgs)
            else:
                # Get the neighbors
                neighbors_ids = classes[i]
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
        dtype=None,
    ):
        """
        See AligningAverager2D, adds:

        :params n_angles: Number of brute force rotations to attempt, defaults 360.
        """
        super().__init__(composite_basis, src, alignment_basis, dtype)

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
            basis_coefficients = self.composite_basis.evaluate_t(
                self.src.images(0, np.inf)
            )

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
        diagnostics=False,
        dtype=None,
    ):
        """
        :param composite_basis:  Basis to be used during class average composition.
        :param src: Source of original images.
        :param alignment_src: Optional, source to be used during class average alignment.
        Must be the same resolution as `src`.
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.__cache = dict()
        self.diagnostics = diagnostics
        self.do_cross_corr_translations = True
        self.alignment_src = alignment_src or src

        # TODO, for accomodating different resolutions we minimally need to adapt shifting.
        # Outside of scope right now, but would make a nice PR later.
        if self.alignment_src.L != src.L:
            raise RuntimeError("Currently `alignment_src.L` must equal `src.L`")
        if self.alignment_src.dtype != src.dtype:
            raise RuntimeError("Currently `alignment_src.dtype` must equal `src.dtype`")

        super().__init__(composite_basis, src, composite_basis, dtype=dtype)

    def _phase_cross_correlation(self, img0, img1):
        """
        # Adapted from skimage.registration.phase_cross_correlation

        :param img0: Fixed image.
        :param img1: Translated image.
        :returns: (cross-correlation magnitudes (2D array), shifts)
        """

        # Cache img0 transform, this saves n_classes*(n_nbor-1) transforms
        # Note we use the `id` because ndarray are unhashable
        key = id(img0)
        if key not in self.__cache:
            self.__cache[key] = fft.fft2(img0)
        src_f = self.__cache[key]

        target_f = fft.fft2(img1)

        # Whole-pixel shifts - Compute cross-correlation by an IFFT
        shape = src_f.shape
        image_product = src_f * target_f.conj()
        cross_correlation = fft.ifft2(image_product)

        # Locate maximum
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), cross_correlation.shape
        )
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

        return np.abs(cross_correlation), shifts

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

        for k in trange(n_classes):
            # # Get the array of images for this class, using the `alignment_src`.
            images = self._cls_images(classes[k], src=self.alignment_src)

            rotations[k], shifts[k], correlations[k] = self._reddychatterji(
                images, classes[k], reflections[k]
            )

        return rotations, shifts, correlations

    def _reddychatterji(self, images, class_k, reflection_k):
        """
        Compute the Reddy Chatterji method registering images[1:] to image[0].

        This differs from papers and published scikit implimentations by
        computing the fixed base image[0] pipeline once then reusing.

        This is a util function to help loop over `classes`.

        :param images: Image data (m_img, L, L)
        :param class_k: Image indices (m_img,)
        :param reflection_k: Image reflections (m_img,)
        :returns: (rotations_k, correlations_k, shifts_k) corresponding to `images`
        """

        # Result arrays
        M = len(images)
        rotations_k = np.zeros(M, dtype=self.dtype)
        correlations_k = np.zeros(M, dtype=self.dtype)
        shifts_k = np.zeros((M, 2), dtype=int)

        # De-Mean, note images is mutated and should be a `copy`.
        images -= images.mean(axis=(-1, -2))[:, np.newaxis, np.newaxis]

        # Precompute fixed_img data used repeatedly in the loop below.
        fixed_img = images[0]
        # Difference of Gaussians (Band Filter)
        fixed_img_dog = difference_of_gaussians(fixed_img, 1, 4)
        # Window Images (Fix spectral boundary)
        wfixed_img = fixed_img_dog * window("hann", fixed_img.shape)
        # Transform image to Fourier space
        fixed_img_fs = np.abs(fft.fftshift(fft.fft2(wfixed_img))) ** 2
        # Compute Log Polar Transform
        radius = fixed_img_fs.shape[0] // 8  # Low Pass
        warped_fixed_img_fs = warp_polar(
            fixed_img_fs,
            radius=radius,
            output_shape=fixed_img_fs.shape,
            scaling="log",
        )
        # Only use half of FFT, because it's symmetrical
        warped_fixed_img_fs = warped_fixed_img_fs[: fixed_img_fs.shape[0] // 2, :]

        # Now prepare for rotating original images,
        #   and searching for translations.
        # We start back at the raw fixed_img.
        twfixed_img = fixed_img * window("hann", fixed_img.shape)

        # Register image `m` against image[0]
        for m in range(1, len(images)):
            # Get the image to register
            regis_img = images[m]

            # Reflect images when necessary
            if reflection_k[m]:
                regis_img = np.flipud(regis_img)

            # Difference of Gaussians (Band Filter)
            regis_img_dog = difference_of_gaussians(regis_img, 1, 4)

            # Window Images (Fix spectral boundary)
            wregis_img = regis_img_dog * window("hann", regis_img.shape)

            self._input_images_diagnostic(
                class_k[0], wfixed_img, class_k[m], wregis_img
            )

            # Transform image to Fourier space
            regis_img_fs = np.abs(fft.fftshift(fft.fft2(wregis_img))) ** 2

            self._windowed_psd_diagnostic(
                class_k[0], fixed_img_fs, class_k[m], regis_img_fs
            )

            # Compute Log Polar Transform
            warped_regis_img_fs = warp_polar(
                regis_img_fs,
                radius=radius,  # Low Pass
                output_shape=fixed_img_fs.shape,
                scaling="log",
            )

            self._log_polar_diagnostic(
                class_k[0], warped_fixed_img_fs, class_k[m], warped_regis_img_fs
            )

            # Only use half of FFT, because it's symmetrical
            warped_regis_img_fs = warped_regis_img_fs[: fixed_img_fs.shape[0] // 2, :]

            # Compute the Cross_Correlation to estimate rotation
            # Note that _phase_cross_correlation uses the mangnitudes (abs()),
            #  ie it is using both freq and phase information.
            cross_correlation, _ = self._phase_cross_correlation(
                warped_fixed_img_fs, warped_regis_img_fs
            )

            # Rotating Cartesian space translates the angular log polar component.
            # Scaling Cartesian space translates the radial log polar component.
            # In common image resgistration problems, both components are used
            #   to simultaneously estimate scaling and rotation.
            # Since we are not currently concerned with scaling transformation,
            #   disregard the second axis of the `cross_correlation` returned by
            #   `_phase_cross_correlation`.
            cross_correlation_score = cross_correlation[:, 0].ravel()

            self._rotation_cross_corr_diagnostic(
                cross_correlation, cross_correlation_score
            )

            # Recover the angle from index representing maximal cross_correlation
            recovered_angle_degrees = (360 / regis_img_fs.shape[0]) * np.argmax(
                cross_correlation_score
            )

            if recovered_angle_degrees > 90:
                r = 180 - recovered_angle_degrees
            else:
                r = -recovered_angle_degrees

            # For now, try the hack below, attempting two cases ...
            # Some papers mention running entire algos /twice/,
            #   when admitting reflections, so this hack is not
            #   the worst you could do :).
            # Hack
            regis_img_estimated = rotate(regis_img, r)
            regis_img_rotated_p180 = rotate(regis_img, r + 180)
            da = np.dot(fixed_img.flatten(), regis_img_estimated.flatten())
            db = np.dot(fixed_img.flatten(), regis_img_rotated_p180.flatten())
            if db > da:
                regis_img_estimated = regis_img_rotated_p180
                r += 180

            self._rotated_diagnostic(
                class_k[0],
                fixed_img,
                class_k[m],
                regis_img_estimated,
                reflection_k[m],
                r,
            )

            # Assign estimated rotations results
            rotations_k[m] = -r * np.pi / 180  # Reverse rot and convert to radians

            if self.do_cross_corr_translations:
                # Prepare for searching over translations using cross-correlation with the rotated image.
                twregis_img = regis_img_estimated * window("hann", regis_img.shape)
                cross_correlation, shift = self._phase_cross_correlation(
                    twfixed_img, twregis_img
                )

                self._translation_cross_corr_diagnostic(cross_correlation)

                # Compute the shifts as integer number of pixels,
                shift_x, shift_y = int(shift[1]), int(shift[0])
                # then apply the shifts
                regis_img_estimated = np.roll(regis_img_estimated, shift_y, axis=0)
                regis_img_estimated = np.roll(regis_img_estimated, shift_x, axis=1)
                # Assign estimated shift to results
                shifts_k[m] = shift[::-1].astype(int)

                self._averaged_diagnostic(
                    class_k[0],
                    fixed_img,
                    class_k[m],
                    regis_img_estimated,
                    reflection_k[m],
                    r,
                )
            else:
                shift = None  # For logger line

            # Estimated `corr` metric
            corr = np.dot(fixed_img.flatten(), regis_img_estimated.flatten())
            correlations_k[m] = corr

            logger.debug(
                f"ref {class_k[0]}, Neighbor {m} Index {class_k[m]}"
                f" Estimates: {r}*, Shift: {shift},"
                f" Corr: {corr}, Refl?: {reflection_k[m]}"
            )

        # Cleanup some cached stuff for this class
        self.__cache.pop(id(warped_fixed_img_fs), None)
        self.__cache.pop(id(twfixed_img), None)

        return rotations_k, shifts_k, correlations_k

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

        rotations, shifts, _ = self.align(classes, reflections, coefs)

        n_classes, n_nbor = classes.shape

        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        for i in tqdm(range(n_classes)):

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
                neighbors_coefs, rotations[i], reflections[i]
            )

            # Note shifts are after rotation for this approach!
            if shifts is not None:
                neighbors_coefs = self.composite_basis.shift(neighbors_coefs, shifts[i])

            # Averaging in composite_basis
            b_avgs[i] = np.mean(neighbors_coefs, axis=0)

        # Now we convert the averaged images from Basis to Cartesian.
        return ArrayImageSource(self.composite_basis.evaluate(b_avgs))

    def _input_images_diagnostic(self, ia, a, ib, b):
        if not self.diagnostics:
            return
        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        ax[0].set_title(f"Image {ia}")
        ax[0].imshow(a)
        ax[1].set_title(f"Image {ib}")
        ax[1].imshow(b)
        plt.show()

    def _windowed_psd_diagnostic(self, ia, a, ib, b):
        if not self.diagnostics:
            return
        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        ax[0].set_title(f"Image {ia} PSD")
        ax[0].imshow(np.log(a))
        ax[1].set_title(f"Image {ib} PSD")
        ax[1].imshow(np.log(b))
        plt.show()

    def _log_polar_diagnostic(self, ia, a, ib, b):
        if not self.diagnostics:
            return
        labels = np.arange(0, 360, 60)
        y = labels / (360 / a.shape[0])

        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        ax[0].set_title(f"Image {ia}")
        ax[0].imshow(a)
        ax[0].set_yticks(y, minor=False)
        ax[0].set_yticklabels(labels)
        ax[0].set_ylabel("Theta (Degrees)")

        ax[1].set_title(f"Image {ib}")
        ax[1].imshow(b)
        ax[1].set_yticks(y, minor=False)
        ax[1].set_yticklabels(labels)
        plt.show()

    def _rotation_cross_corr_diagnostic(
        self, cross_correlation, cross_correlation_score
    ):
        if not self.diagnostics:
            return
        labels = [0, 30, 60, 90, -60, -30]
        x = y = np.arange(0, 180, 30) / (180 / cross_correlation.shape[0])
        plt.title("Rotation Cross Correlation Map")
        plt.imshow(cross_correlation)
        plt.xlabel("Scale")
        plt.yticks(y, labels, rotation="vertical")
        plt.ylabel("Theta (Degrees)")
        plt.show()

        plt.plot(cross_correlation_score)
        plt.title("Angle vs Cross Correlation Score")
        plt.xticks(x, labels)
        plt.xlabel("Theta (Degrees)")
        plt.ylabel("Cross Correlation Score")
        plt.grid()
        plt.show()

    def _rotated_diagnostic(self, ia, a, ib, b, sb, rb):
        """
        Plot the image after estimated rotation and reflection.

        :param ia: index image `a`
        :param a: image `a`
        :param ib: index image `b`
        :param b: image `b` after reflection `sb` and rotion `rb`
        :param sb: Reflection, Boolean
        :param rb: Estimated rotation, degrees
        """

        if not self.diagnostics:
            return

        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        ax[0].set_title(f"Image {ia}")
        ax[0].imshow(a)
        ax[0].grid()
        ax[1].set_title(f"Image {ib} Refl: {str(sb)[0]} Rotated {rb:.1f}")
        ax[1].imshow(b)
        ax[1].grid()
        plt.show()

    def _translation_cross_corr_diagnostic(self, cross_correlation):
        if not self.diagnostics:
            return
        plt.title("Translation Cross Correlation Map")
        plt.imshow(cross_correlation)
        plt.xlabel("x shift (pixels)")
        plt.ylabel("y shift (pixels)")
        L = self.alignment_src.L
        labels = [0, 10, 20, 30, 0, -10, -20, -30]
        tick_location = [0, 10, 20, 30, L, L - 10, L - 20, L - 30]
        plt.xticks(tick_location, labels)
        plt.yticks(tick_location, labels)
        plt.show()

    def _averaged_diagnostic(self, ia, a, ib, b, sb, rb):
        """
        Plot the stacked average image after
        estimated rotation and reflections.

        Compare in a three way plot.

        :param ia: index image `a`
        :param a: image `a`
        :param ib: index image `b`
        :param b: image `b` after reflection `sb` and rotion `rb`
        :param sb: Reflection, Boolean
        :param rb: Estimated rotation, degrees
        """
        if not self.diagnostics:
            return
        fig, axes = plt.subplots(1, 3)
        ax = axes.ravel()
        ax[0].set_title(f"{ia}")
        ax[0].imshow(a)
        ax[0].grid()
        ax[1].set_title(f"{ib} Refl: {str(sb)[0]} Rot: {rb:.1f}")
        ax[1].imshow(b)
        ax[1].grid()
        ax[2].set_title("Stacked Avg")
        plt.imshow((a + b) / 2.0)
        ax[2].grid()
        plt.show()


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
        diagnostics=False,
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

        :param diagnostics: Plot interactive diagnostic graphics (for debugging).
        :param dtype: Numpy dtype to be used during alignment.
        """

        super().__init__(
            composite_basis,
            src,
            alignment_src,
            diagnostics,
            dtype=dtype,
        )

        # For brute force we disable the cross_corr translation code
        self.do_cross_corr_translations = False
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
        _rotations = np.zeros(classes.shape, dtype=self.dtype)
        rotations = np.zeros(classes.shape, dtype=self.dtype)
        _correlations = np.zeros(classes.shape, dtype=self.dtype)
        correlations = np.ones(classes.shape, dtype=self.dtype) * -np.inf
        shifts = np.zeros((*classes.shape, 2), dtype=int)

        # We'll brute force all shifts in a grid.
        g = grid_2d(L, normalized=False)
        disc = g["r"] <= self.radius
        X, Y = g["x"][disc], g["y"][disc]

        for k in trange(n_classes):
            unshifted_images = self._cls_images(classes[k])

            for xs, ys in zip(X, Y):
                s = np.array([xs, ys])
                # Get the array of images for this class

                # Note we mutate `images` here with shifting,
                #   then later in `_reddychatterji`
                images = unshifted_images.copy()
                # Don't shift the base image
                images[1:] = Image(unshifted_images[1:]).shift(s).asnumpy()

                rotations[k], _, correlations[k] = self._reddychatterji(
                    images, classes[k], reflections[k]
                )

                # Where corr has improved
                #  update our rolling best results with this loop.
                improved = _correlations > correlations
                correlations = np.where(improved, _correlations, correlations)
                rotations = np.where(improved, _rotations, rotations)
                shifts = np.where(improved[..., np.newaxis], s, shifts)
                logger.debug(f"Shift {s} has improved {np.sum(improved)} results")

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
