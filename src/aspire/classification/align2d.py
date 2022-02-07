import logging
from abc import ABC, abstractmethod
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import difference_of_gaussians, window

# import skimage.io
from skimage.transform import rotate, warp_polar
from tqdm import tqdm, trange

from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.utils.coor_trans import grid_2d

logger = logging.getLogger(__name__)


class Align2D(ABC):
    """
    Base class for 2D Image Alignment methods.
    """

    def __init__(
        self, alignment_basis, source, composite_basis=None, batch_size=512, dtype=None
    ):
        """
        :param alignment_basis: Basis to be used during alignment (eg FSPCA)
        :param source: Source of original images.
        :param composite_basis:  Basis to be used during class average composition (eg hi res Cartesian/FFB2D)
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.alignment_basis = alignment_basis
        # if composite_basis is None, use alignment_basis
        self.composite_basis = composite_basis or self.alignment_basis
        self.src = source
        self.batch_size = batch_size
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
    def align(self, classes, reflections, basis_coefficients):
        """
        Any align2D alignment method should take in the following arguments
        and return aligned images.

        During this process `rotations`, `reflections`, `shifts` and
        `correlations` properties will be computed for aligners
        that implement them.

        `rotations` is an (n_classes, n_nbor) array of angles,
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

    def _images(self, cls, src=None):
        """
        Util to return images as an array for class k (provided as array `cls` ),
        preserving the class/nbor order.

        :param cls: An iterable (0/1-D array or list) that holds the indices of images to align.
        In Class Averaging, this would be a class.
        :param src: Optionally overridee the src, for example, if you want to use a different
        source for a certain operation (ie aignment).
        """
        src = src or self.src

        n_nbor = cls.shape[-1]  # Includes zero'th neighbor

        images = np.empty((n_nbor, src.L, src.L), dtype=self.dtype)

        for i, index in enumerate(cls):
            images[i] = src.images(index, 1).asnumpy()

        return images


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

        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        for i in tqdm(range(n_classes)):

            # Get coefs in Composite_Basis if not provided as an argumen.
            if coefs is None:
                # Retrieve relavent images directly from source.
                neighbors_imgs = Image(self._images(classes[i]))

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


class BFRAlign2D(AveragedAlign2D):
    """
    This perfoms a Brute Force Rotational alignment.

    For each class,
        constructs n_angles rotations of all class members,
        and then identifies angle yielding largest correlation(dot).
    """

    def __init__(
        self,
        alignment_basis,
        source,
        composite_basis=None,
        n_angles=359,
        batch_size=512,
        dtype=None,
    ):
        """
        :params alignment_basis: Basis providing a `rotate` method.
        :param source: Source of original images.
        :params n_angles: Number of brute force rotations to attempt, defaults 359.
        """
        super().__init__(alignment_basis, source, composite_basis, batch_size, dtype)

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
        batch_size=512,
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
        super().__init__(
            alignment_basis,
            source,
            composite_basis,
            n_angles,
            batch_size=batch_size,
            dtype=dtype,
        )

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
            logger.debug(f"Computing Rotational alignment after shift ({x},{y}).")

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
                logger.debug("Initial rotational alignment complete (shift (0,0))")
                assert np.sum(improved_indices) == np.size(
                    classes
                ), f"{np.sum(improved_indices)} =?= {np.size(classes)}"
            else:
                logger.debug(
                    f"Shift ({x},{y}) complete. Improved {np.sum(improved_indices)} alignments."
                )

        return classes, reflections, rotations, shifts, correlations


class ReddyChatterjiAlign2D(AveragedAlign2D):
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
        alignment_basis,
        source,
        composite_basis=None,
        alignment_source=None,
        diagnostics=False,
        batch_size=512,
        dtype=None,
    ):
        """
        :param alignment_basis: Basis to be used during alignment.
        For current implementation of ReddyChatterjiAlign2D this should be `None`.
        Instead see `alignment_source`.
        :param source: Source of original images.
        :param composite_basis:  Basis to be used during class average composition.
        For current implementation of ReddyChatterjiAlign2D this should be `None`.
        Instead this method uses `source` for composition of the averaged stack.
        :param alignment_source:  Basis to be used during class average composition.
        Must be the same resolution as `source`.
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.__cache = dict()
        self.diagnostics = diagnostics
        self.do_cross_corr_translations = True
        self.alignment_src = alignment_source or source

        # TODO, for accomodating different resolutions we minimally need to adapt shifting.
        # Outside of scope right now, but would make a nice PR later.
        if self.alignment_src.L != source.L:
            raise RuntimeError("Currently `alignment_src.L` must equal `source.L`")
        if self.alignment_src.dtype != source.dtype:
            raise RuntimeError(
                "Currently `alignment_src.dtype` must equal `source.dtype`"
            )

        # Sanity check. This API should be rethought once all basis and
        # alignment methods have been incorporated.
        assert alignment_basis is None  # We use sources directly for alignment
        assert (
            composite_basis is not None
        )  # However, we require a basis for rotating etc.

        super().__init__(
            alignment_basis, source, composite_basis, batch_size=batch_size, dtype=dtype
        )

    def _phase_cross_correlation(self, img0, img1):
        """
        # Adapted from skimage.registration.phase_cross_correlation

        :param img0: Fixed image.
        :param img1: Translated image.
        :returns: (cross-correlation magnitudes (2D array), shifts)
        """

        # Cache img0 transform, this saves n_classes*(n_nbor-1) transforms
        # Note we use the `id` because ndarray are unhashable
        src_f = self.__cache.setdefault(id(img0), np.fft.fft2(img0))

        target_f = np.fft.fft2(img1)

        # Whole-pixel shifts - Compute cross-correlation by an IFFT
        shape = src_f.shape
        image_product = src_f * target_f.conj()
        cross_correlation = np.fft.ifft2(image_product)

        # Locate maximum
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), cross_correlation.shape
        )
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

        return np.abs(cross_correlation), shifts

    def _align(self, classes, reflections, basis_coefficients):
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
            images = self._images(classes[k], src=self.alignment_src)

            self._reddychatterji(
                k, images, classes, reflections, rotations, correlations, shifts
            )

        return classes, reflections, rotations, shifts, correlations

    def _reddychatterji(
        self, k, images, classes, reflections, rotations, correlations, shifts
    ):
        """
        Compute the Reddy Chatterji registering  images[1:] to image[0].

        This differs from papers and published scikit implimentations by
        computing the fixed base image[0] pipeline once then reusing.
        """

        # De-Mean
        images -= images.mean(axis=(-1, -2))[:, np.newaxis, np.newaxis]

        # Precompute fixed_img data used repeatedly in the loop below.
        fixed_img = images[0]
        # Difference of Gaussians (Band Filter)
        fixed_img_dog = difference_of_gaussians(fixed_img, 1, 4)
        # Window Images (Fix spectral boundary)
        wfixed_img = fixed_img_dog * window("hann", fixed_img.shape)
        # Transform image to Fourier space
        fixed_img_fs = np.abs(np.fft.fftshift(np.fft.fft2(wfixed_img))) ** 2
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

            # Reflect images when nessecary
            if reflections[k][m]:
                regis_img = np.flipud(regis_img)

            # Difference of Gaussians (Band Filter)
            regis_img_dog = difference_of_gaussians(regis_img, 1, 4)

            # Window Images (Fix spectral boundary)
            wregis_img = regis_img_dog * window("hann", regis_img.shape)

            self._input_images_diagnostic(
                classes[k][0], wfixed_img, classes[k][m], wregis_img
            )

            # Transform image to Fourier space
            regis_img_fs = np.abs(np.fft.fftshift(np.fft.fft2(wregis_img))) ** 2

            self._windowed_psd_diagnostic(
                classes[k][0], fixed_img_fs, classes[k][m], regis_img_fs
            )

            # Compute Log Polar Transform
            warped_regis_img_fs = warp_polar(
                regis_img_fs,
                radius=radius,  # Low Pass
                output_shape=fixed_img_fs.shape,
                scaling="log",
            )

            self._log_polar_diagnostic(
                classes[k][0], warped_fixed_img_fs, classes[k][m], warped_regis_img_fs
            )

            # Only use half of FFT, because it's symmetrical
            warped_regis_img_fs = warped_regis_img_fs[: fixed_img_fs.shape[0] // 2, :]

            # Compute the Cross_Correlation to estimate rotation
            # Note that _phase_cross_correlation uses the mangnitudes (abs()),
            #  ie it is using both freq and phase information.
            cross_correlation, shift = self._phase_cross_correlation(
                warped_fixed_img_fs, warped_regis_img_fs
            )

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

            # Dont like this, but I got stumped/frustrated.
            # For now, try the hack below, attempting two cases ...
            # Most of the papers mention running the whole algo /twice/,
            #   when admitting reflections, so this hack is not
            #   the worst you could do :).
            # if reflections[k][m]:
            #     if 0<= r < 90:
            #         r -= 180
            # Hack
            regis_img_estimated = rotate(regis_img, r)
            regis_img_rotated_p180 = rotate(regis_img, r + 180)
            da = np.dot(fixed_img.flatten(), regis_img_estimated.flatten())
            db = np.dot(fixed_img.flatten(), regis_img_rotated_p180.flatten())
            if db > da:
                regis_img_estimated = regis_img_rotated_p180
                r += 180

            self._rotated_diagnostic(
                classes[k][0],
                fixed_img,
                classes[k][m],
                regis_img_estimated,
                reflections[k][m],
                r,
            )

            # Assign estimated rotations results
            rotations[k][m] = -r * np.pi / 180  # Reverse rot and convert to radians

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
                shifts[k][m] = shift[::-1].astype(int)

                self._averaged_diagnostic(
                    classes[k][0],
                    fixed_img,
                    classes[k][m],
                    regis_img_estimated,
                    reflections[k][m],
                    r,
                )
            else:
                shift = None  # For logger line

            # Estimated `corr` metric
            corr = np.dot(fixed_img.flatten(), regis_img_estimated.flatten())
            correlations[k][m] = corr

            logger.debug(
                f"Class {k}, ref {classes[k][0]}, Neighbor {m} Index {classes[k][m]}"
                f" Estimates: {r}*, Shift: {shift},"
                f" Corr: {corr}, Refl?: {reflections[k][m]}"
            )

        # Cleanup some cached stuff for this class
        self.__cache.pop(id(warped_fixed_img_fs), None)
        self.__cache.pop(id(twfixed_img), None)

    def average(
        self,
        classes,
        reflections,
        rotations,
        shifts=None,
        coefs=None,
    ):
        """
        This averages classes performing rotations then shifts.
        Otherwise is similar to `AveragedAlign2D.average`.
        """
        n_classes, n_nbor = classes.shape

        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        for i in tqdm(range(n_classes)):

            # Get coefs in Composite_Basis if not provided as an argument.
            if coefs is None:
                # Retrieve relavent images directly from source.
                neighbors_imgs = Image(self._images(classes[i]))
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


class BFSReddyChatterjiAlign2D(ReddyChatterjiAlign2D):
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
        alignment_basis,
        source,
        composite_basis=None,
        alignment_source=None,
        radius=None,
        diagnostics=False,
        batch_size=512,
        dtype=None,
    ):
        """
        :param alignment_basis: Basis to be used during alignment.
        For current implementation of ReddyChatterjiAlign2D this should be `None`.
        Instead see `alignment_source`.
        :param source: Source of original images.
        :param composite_basis:  Basis to be used during class average composition.
        For current implementation of ReddyChatterjiAlign2D this should be `None`.
        Instead this method uses `source` for composition of the averaged stack.
        :param alignment_source:  Basis to be used during class average composition.
        Must be the same resolution as `source`.
        :param radius: Brute force translation search radius.
        Defaults to source.L//8.
        :param dtype: Numpy dtype to be used during alignment.

        :param diagnostics: Plot interactive diagnostic graphics (for debugging).
        :param dtype: Numpy dtype to be used during alignment.
        """

        super().__init__(
            alignment_basis,
            source,
            composite_basis,
            alignment_source,
            diagnostics,
            batch_size=batch_size,
            dtype=dtype,
        )

        # For brute force we disable the cross_corr translation code
        self.do_cross_corr_translations = False
        # Assign search radius
        self.radius = radius or source.L // 8

    def _align(self, classes, reflections, basis_coefficients):
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
        _shifts = np.zeros((*classes.shape, 2), dtype=int)
        shifts = np.zeros((*classes.shape, 2), dtype=int)

        # We'll brute force all shifts in a grid.
        g = grid_2d(L, normalized=False)
        disc = g["r"] <= L // 8  # make param later
        X, Y = g["x"][disc], g["y"][disc]

        for k in trange(n_classes):
            unshifted_images = self._images(classes[k])

            for xs, ys in zip(X, Y):
                s = np.array([xs, ys])
                # Get the array of images for this class

                images = unshifted_images.copy()
                # Don't shift the base image
                images[1:] = Image(unshifted_images[1:]).shift(s).asnumpy()

                self._reddychatterji(
                    k, images, classes, reflections, _rotations, _correlations, _shifts
                )

                # Where corr has improved
                #  update our rolling best results with this loop.
                improved = _correlations > correlations
                correlations = np.where(improved, _correlations, correlations)
                rotations = np.where(improved, _rotations, rotations)
                shifts = np.where(shifts, _shifts, shifts)
                logger.debug(f"Shift {s} has improved {np.sum(improved)} results")

        return classes, reflections, rotations, shifts, correlations

    def average(
        self,
        classes,
        reflections,
        rotations,
        shifts=None,
        coefs=None,
    ):
        """
        See AveragedAlign2D.average.
        """
        # ReddyChatterjiAlign2D does rotations then shifts.
        # For brute force, we'd like shifts then rotations,
        #   as is done in gerneral via AveragedAlign2D.
        return AveragedAlign2D.average(
            self, classes, reflections, rotations, shifts, coefs
        )


class EMAlign2D(Align2D):
    """
    Citation needed.
    """


class FTKAlign2D(Align2D):
    """
    Factorization of the translation kernel for fast rigid image alignment.
    Rangan, A.V., Spivak, M., Anden, J., & Barnett, A.H. (2019).
    """
