import logging
from abc import ABC, abstractmethod

import numpy as np

from aspire.basis import Coef
from aspire.classification.reddy_chatterji import reddy_chatterji_register
from aspire.image import Image, ImageStacker, MeanImageStacker
from aspire.numeric import fft, xp
from aspire.operators import PolarFT
from aspire.utils import complex_type, tqdm, trange
from aspire.utils.coor_trans import grid_2d

logger = logging.getLogger(__name__)


def commute_shift_rot(shifts, rots):
    """
    Rotate `shifts` points by `rots` ccw radians.

    :param shifts: Array of shift points shaped (..., 2)
    :param rots: Array of rotations (radians)
    :returns: Array of rotated shift points shaped (..., 2)
    """
    sx = shifts[:, 0]
    sy = shifts[:, 1]
    x = sx * np.cos(rots) - sy * np.sin(rots)
    y = sx * np.sin(rots) + sy * np.cos(rots)
    return np.stack((x, y), axis=1)


class Averager2D(ABC):
    """
    Base class for 2D Image Averaging methods.
    """

    def __init__(self, composite_basis, src, batch_size=512, dtype=None):
        """
        :param composite_basis:  Basis to be used during class average composition (eg FFB2D)
        :param src: Source of original images.
        :param batch_size: Integer size of batches used for basis conversion.
        :param dtype: Numpy dtype to be used during alignment.
        """

        self.composite_basis = composite_basis
        self.src = src
        self.batch_size = int(batch_size)

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

        :param classes: class indices, refering to src. (src.n, n_nbor).
        :param reflections: Bool representing whether to reflect image in `classes`.
            (n_clases, n_nbor)
        :param coefs: Optional basis coefs (could avoid recomputing).
            (src.n, coef_count)
        :return: Stack of synthetic class average images as Image instance.
        """

    def _cls_images(self, cls, src=None):
        """
        Util to return images as an array for class k (provided as
        array `cls` ), preserving the class/nbor order.  Note, Images
        will be a read-only view, copy if mutations required.

        :param cls: An iterable (0/1-D array or list) that holds the indices of images to align.
            In class averaging, this would be a class.
        :param src: Optionally override the src, for example, if you want to use a different
            source for a certain operation (ie alignment).
        """
        src = src or self.src
        return src.images[cls].asnumpy().astype(self.dtype, copy=False)


class AligningAverager2D(Averager2D):
    """
    Subclass supporting averagers which perfom an aligning stage.
    """

    def __init__(
        self,
        composite_basis,
        src,
        alignment_basis=None,
        image_stacker=None,
        batch_size=512,
        dtype=None,
    ):
        """
        :param composite_basis:  Basis to be used during class average composition (eg hi res Cartesian/FFB2D).
        :param src: Source of original images.
        :param alignment_basis: Optional, basis to be used only during alignment (eg FSPCA).
        :param image_stacker: Optional, provide a user defined `ImageStacker` instance,
            used during image stacking (averaging).  Defaults to MeanImageStacker.
        :param batch_size: Integer size of batches used for basis conversion.
        :param dtype: Numpy dtype to be used during alignment.
        """

        super().__init__(
            composite_basis=composite_basis,
            src=src,
            batch_size=batch_size,
            dtype=dtype,
        )
        # If alignment_basis is None, use composite_basis
        self.alignment_basis = alignment_basis or self.composite_basis

        # If image_stacker is None, use mean
        self.image_stacker = image_stacker or MeanImageStacker()
        if not isinstance(self.image_stacker, ImageStacker):
            raise ValueError("`image_stacker` should be subclass of ImageStacker.")

        if not hasattr(self.composite_basis, "rotate"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s composite_basis {self.composite_basis} must provide a `rotate` method."
            )
        if not hasattr(self.composite_basis, "shift"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s composite_basis {self.composite_basis} must provide a `shift` method."
            )

    @abstractmethod
    def align(self, classes, reflections, basis_coefficients=None):
        """
        During this process `rotations`, `reflections`, `shifts` and
        `dot_products` properties will be computed for aligners.

        `rotations` is an (src.n, n_nbor) array of angles,
        which should represent the rotations needed to align images within
        that class. `rotations` is measured in CCW radians.

        `shifts` is None or an (src.n, n_nbor) array of 2D shifts
        which should represent the translation needed to best align the images
        within that class.

        `dot_products` is an (src.n, n_nbor) array representing
        the dot between classified images and their base
        image (image index 0).

        Subclasses of should implement and extend this method.

        :param classes: (src.n, n_nbor) integer array of img indices.
        :param reflections: (src.n, n_nbor) bool array of corresponding reflections,
        :param basis_coefficients: (n_img, self.alignment_basis.count) basis coefficients,

        :returns: (rotations, shifts, dot_products)
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

        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        self.rotations, self.shifts, self.dot_products = self.align(
            classes, reflections, coefs
        )

        n_classes, n_nbor = classes.shape

        # Result (image) array
        avgs = np.empty((n_classes, *self.composite_basis.sz), dtype=self.src.dtype)
        # Tmp (basis) batch result array)
        b_avgs = np.empty(
            (self.batch_size, self.composite_basis.count), dtype=self.src.dtype
        )

        def _innerloop(i):
            # Get coefs in Composite_Basis if not provided as an argument.
            if coefs is None:
                # Retrieve relevant images directly from source.
                neighbors_imgs = Image(self._cls_images(classes[i]))

                # Do shifts
                if self.shifts is not None:
                    neighbors_imgs = neighbors_imgs.shift(self.shifts[i])

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
            return self.image_stacker(neighbors_coefs.asnumpy())

        desc = f"Stacking and evaluating class averages from {self.composite_basis.__class__.__name__} to Cartesian"
        for start in trange(0, n_classes, self.batch_size, desc=desc):
            end = min(start + self.batch_size, n_classes)
            for i, cls in enumerate(
                trange(start, end, desc="Stacking batch", leave=False)
            ):
                b_avgs[i] = _innerloop(cls)  # average stacked in basis

            # Now we convert the averaged images from Basis to Cartesian,
            #   assigning to result array.
            # Note i should usually be batch_size, but may be less on final batch.
            avgs[start:end] = (
                Coef(self.composite_basis, b_avgs[: i + 1]).evaluate().asnumpy()
            )

        return Image(avgs)

    def _shift_search_grid(self, L, radius, roll_zero=False, sub_pixel=1):
        """
        Returns two 1-D arrays representing the X and Y grid points in the defined
        shift search space (disc <= self.radius).

        :param radius: Disc radius in pixels
        :param roll_zero: Roll (0,0) to zero'th element. Defaults to False.
        :param sub_pixel: Sub-pixel decimation .  1 yields 1 pixel, 10 yields 1/10 pixel, etc.
            Values will be cast to integers.
        :returns: Grid points as array of 2-tuples [(x0,y0),... (xi,yi)].
        """
        sub_pixel = int(sub_pixel)

        # We'll brute force all shifts in a grid.
        g = grid_2d(sub_pixel * L, normalized=False)
        disc = g["r"] <= (sub_pixel * radius)
        X, Y = g["x"][disc], g["y"][disc]
        X, Y = X / sub_pixel, Y / sub_pixel

        # Optionally roll arrays so 0 is first.
        if roll_zero:
            zero_ind = np.argwhere(X * X + Y * Y == 0).flatten()[0]
            X, Y = np.roll(X, -zero_ind), np.roll(Y, -zero_ind)
            assert (X[0], Y[0]) == (0, 0), (radius, zero_ind, X, Y)

        shifts = np.stack((X, Y), axis=1)

        return shifts


class BFSRAverager2D(AligningAverager2D):
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
        radius=None,
        batch_size=512,
        dtype=None,
    ):
        """
        See AligningAverager2D adds `n_angles` and `radius`.

        :params n_angles: Number of brute force rotations to attempt, defaults 360.
        :param radius: Brute force translation search radius.
            Defaults to src.L//32.
        """
        super().__init__(
            composite_basis,
            src,
            alignment_basis,
            batch_size=batch_size,
            dtype=dtype,
        )

        self.n_angles = n_angles

        if not hasattr(self.alignment_basis, "rotate"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s alignment_basis {self.alignment_basis} must provide a `rotate` method."
            )

        self.radius = radius if radius is not None else src.L // 32

        if self.radius != 0:

            if not hasattr(self.alignment_basis, "shift"):
                raise RuntimeError(
                    f"{self.__class__.__name__}'s alignment_basis {self.alignment_basis} must provide a `shift` method."
                )

    def align(self, classes, reflections, basis_coefficients=None):
        """
        See `AligningAverager2D.align`
        """

        # Admit simple case of single case alignment
        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        # Result arrays
        # These arrays will incrementally store our best alignment.
        n_classes, n_nbor = classes.shape
        rotations = np.zeros((n_classes, n_nbor), dtype=self.dtype)
        dot_products = np.ones((n_classes, n_nbor), dtype=self.dtype) * -np.inf
        shifts = np.empty((*classes.shape, 2), dtype=int)

        # Work arrays
        _rotations = np.zeros((n_nbor), dtype=self.dtype)
        _dot_products = np.ones((n_nbor), dtype=self.dtype) * -np.inf

        # Construct array of angles to brute force.
        _angles = xp.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)

        ks = xp.asarray(self.alignment_basis.complex_angular_indices).reshape(-1, 1)
        _rot_ops_conj = xp.exp(1j * ks * _angles).conj()

        # Create a search grid and force initial pair to (0,0)
        # This is done primarily in case of a tie later, we would take unshifted.
        test_shifts = self._shift_search_grid(self.src.L, self.radius, roll_zero=True)

        for k in trange(n_classes, desc="Rotationally aligning classes"):
            # We want to locally cache the original images,
            #  because we will mutate them with shifts in the next loop.
            #  This avoids recomputing them before each shift
            # The coefficient for the base images are also computed here.
            if basis_coefficients is None:
                original_images = Image(self._cls_images(classes[k], src=self.src))
                _coef0 = self.alignment_basis.evaluate_t(original_images[0])
            else:
                original_coef = basis_coefficients[classes[k], :]
                original_images = self.alignment_basis.evaluate(original_coef)
                _coef0 = original_coef[0]

            # Working copy
            _images = original_images.asnumpy().copy()

            # Generate table of rotations for image 0.
            # Note we invert the rotations later.
            #   Applying rot to image 0
            #   avoids rotating each member of the class
            #   for the argmax alignment test.
            # Convert to array of complex coef, implicit copy.
            _coef0 = xp.array(_coef0.to_complex().asnumpy())
            base_img = _coef0.reshape(self.alignment_basis.complex_count, 1)
            # (cnt, n_rot) * (cnt, 1) -> (cnt, n_rot)
            rot_base_imgs_conj = _rot_ops_conj * base_img.conj()

            # Loop over shift search space, updating best result
            for x, y in tqdm(
                test_shifts,
                total=len(test_shifts),
                desc="\tmaximizing over shifts",
                disable=len(test_shifts) == 1,
                leave=False,
            ):
                shift = np.array([x, y], dtype=int)
                logger.debug(f"Computing rotational alignment after shift ({x},{y}).")

                # For each shift, the set of neighbor images is shifted.
                #   This order is chosen because:
                #   i) allows concatenation of shifts and rotation
                #   operations after orientation estimation
                #   ii) because generally the number of neighbors << the
                #   number of test rotations.

                # Skip zero shifting.
                if np.any(shift != 0):
                    # Note the base image[0] is never shifted.
                    _images[1:] = original_images[1:].shift(shift).asnumpy()

                # Convert to array of complex coef, implicit copy.
                _coef = self.alignment_basis.evaluate_t(Image(_images))
                _coef = xp.array(_coef.to_complex().asnumpy())

                # Handle reflections
                refl = reflections[k]
                _coef[refl] = xp.conj(_coef[refl])

                # Compute dot product of each base-neighbor pair.
                #   The collection of dots is performed in bulk
                #   as a large matmul.
                # (n_nbor, cnt) @ (cnt, n_rot) = (n_nbor, n_rot)
                dots = xp.real(_coef @ rot_base_imgs_conj)
                idx = xp.argmax(dots, axis=1)
                idx[0] = 0  # Force base image, just in case.

                # Assign results for this class
                _dot_products[:] = xp.asnumpy(
                    xp.take_along_axis(dots, idx.reshape(n_nbor, 1), axis=1).flatten()
                )
                # Note, legacy codes would normalize to form correlations.
                #   These were only used for diagnostic purposes.
                #   Normalizing is skipped here to save computation.

                # Assign the reverse rotation
                _rotations[:] = -1 * xp.asnumpy(_angles[idx])

                # Test and update
                # Each base-neighbor pair may have a best shift+rot from a different shift iteration.
                improved_indices = _dot_products > dot_products[k]
                rotations[k, improved_indices] = _rotations[improved_indices]
                dot_products[k, improved_indices] = _dot_products[improved_indices]
                shifts[k, improved_indices] = shift

                if (x, y) == (0, 0):
                    logger.debug("Initial rotational alignment complete (shift (0,0))")
                    assert np.sum(improved_indices) == np.size(
                        classes[0]
                    ), f"{np.sum(improved_indices)} =?= {np.size(classes)}"
                else:
                    logger.debug(
                        f"Shift ({x},{y}) complete. Improved {np.sum(improved_indices)} alignments."
                    )

        return rotations, shifts, dot_products


class BFRAverager2D(BFSRAverager2D):
    """
    Brute Force Rotation only reference implementation.

    See BFT with `radius=0` for a more performant implementation using a fast rotational alignment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, radius=0, **kwargs)

    def align(self, *args, **kwargs):
        """
        See `BFSRAverager2D.align`
        """
        # BFR shifts should all be zeros.
        # Replace with `None` to induce short ciruit shifting during stacking.
        rotations, shifts, dot_products = super().align(*args, **kwargs)

        # Sanity check the results did not indicate shifts.
        if not np.all(shifts.flatten() == 0):
            raise RuntimeError(
                "BFR should return zero shifts." "  BFSR returned non zero shifts."
            )

        return rotations, None, dot_products


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
        batch_size=512,
        dtype=None,
    ):
        """
        :param composite_basis:  Basis to be used during class average composition.
        :param src: Source of original images.
        :param alignment_src: Optional, source to be used during class average alignment.
            Must be the same resolution as `src`.
        :param batch_size: Integer size of batches used for basis conversion.
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
            composite_basis, src, composite_basis, batch_size=batch_size, dtype=dtype
        )

    def align(self, classes, reflections, basis_coefficients=None):
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
        dot_products = np.zeros(classes.shape, dtype=self.dtype)
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

        for k in trange(n_classes, desc="Rotationally aligning classes"):
            rotations[k], shifts[k], dot_products[k] = _innerloop(k)

        return rotations, shifts, dot_products

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

        self.rotations, self.shifts, self.dot_products = self.align(
            classes, reflections, coefs
        )

        n_classes, n_nbor = classes.shape

        b_avgs = np.empty((n_classes, self.composite_basis.count), dtype=self.src.dtype)

        def _innerloop(i):
            # Get coefs in Composite_Basis if not provided as an argument.
            if coefs is None:
                # Retrieve relevant images directly from source.
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
            return self.image_stacker(neighbors_coefs.asnumpy())

        for i in trange(n_classes, desc="Stacking class averages"):
            b_avgs[i] = _innerloop(i)

        # Now we convert the averaged images from Basis to Cartesian.
        return Coef(self.composite_basis, b_avgs).evaluate()


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
        batch_size=512,
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
        :param batch_size: Integer size of batches used for basis conversion.
        :param dtype: Numpy dtype to be used during alignment.
        """

        super().__init__(
            composite_basis,
            src,
            alignment_src,
            batch_size=batch_size,
            dtype=dtype,
        )

        # Assign search radius
        self.radius = radius if radius is not None else src.L // 8

    def align(self, classes, reflections, basis_coefficients=None):
        """
        Performs the actual rotational alignment estimation,
        returning parameters needed for averaging.
        """

        # Admit simple case of single case alignment
        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        n_classes, n_nbor = classes.shape

        # Instantiate matrices for inner loop, and best results.
        rotations = np.zeros(classes.shape, dtype=self.dtype)
        dot_products = np.ones(classes.shape, dtype=self.dtype) * -np.inf
        shifts = np.zeros((*classes.shape, 2), dtype=int)

        test_shifts = self._shift_search_grid(self.alignment_src.L, self.radius)

        def _innerloop(k):
            unshifted_images = self._cls_images(classes[k])
            # Instantiate matrices for inner loop, and best results.
            _rotations = np.zeros(classes.shape[1:], dtype=self.dtype)
            _dot_products = np.ones(classes.shape[1:], dtype=self.dtype) * -np.inf
            _shifts = np.zeros((*classes.shape[1:], 2), dtype=int)

            for xs, ys in tqdm(
                test_shifts,
                total=len(test_shifts),
                desc="\tmaximizing over shifts",
                disable=len(test_shifts) == 1,
                leave=False,
            ):

                s = np.array([xs, ys])
                # Get the array of images for this class

                # Note we mutate `images` here with shifting,
                #   then later reddy_chatterji_register
                images = unshifted_images.copy()
                # Don't shift the base image
                images[1:] = Image(unshifted_images[1:]).shift(s).asnumpy()

                # returned shifts ignored since we are forcing shift of `s` above
                __rotations, _, __dot_products = reddy_chatterji_register(
                    images,
                    reflections[k],
                    mask=self.mask,
                    do_cross_corr_translations=False,  # When forcing s, we skip cross corr translations
                    dtype=self.dtype,
                )

                # Where corr has improved
                #  update our rolling best results with this loop.
                improved = __dot_products > _dot_products
                _dot_products = np.where(improved, __dot_products, _dot_products)
                _rotations = np.where(improved, __rotations, _rotations)
                _shifts = np.where(improved[..., np.newaxis], s, _shifts)
                logger.debug(f"Shift {s} has improved {np.sum(improved)} results")

            return _rotations, _shifts, _dot_products

        for k in trange(n_classes, desc="Rotationally aligning classes"):
            rotations[k], shifts[k], dot_products[k] = _innerloop(k)

        return rotations, shifts, dot_products

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


class BFTAverager2D(AligningAverager2D):
    """
    This perfoms a Brute Force Translations and fast rotational alignment.

    For each shift,
       Perform polar Fourier cross correlation based rotational alignment.

    Return the rotation and shift yielding the best results.
    """

    def __init__(
        self,
        composite_basis,
        src,
        alignment_basis=None,
        n_angles=360,
        n_radial=None,
        radius=None,
        sub_pixel=1,
        batch_size=512,
        dtype=None,
    ):
        """
        See AligningAverager2D. Adds `n_angles`, `n_radial`, `radius`, `sub_pixel`.

        :params n_angles: Number of PFT angular components, defaults 360.
        :param n_radial: Number of PFT radial components, defaults `self.src.L//2`.
        :param radius: Brute force translation search radius.
            `0` disables translation search, rotations only.
            Defaults to `src.L//32`.
        :param sub_pixel: Sub-pixel decimation used in brute force shift search.
            Defaults to 10 sub-pixel to pixel, ie 0.1 spaced sub-pixel.
        """
        super().__init__(
            composite_basis,
            src,
            alignment_basis,
            batch_size=batch_size,
            dtype=dtype,
        )

        self.n_angles = n_angles

        self.radius = radius if radius is not None else src.L // 32

        if self.radius != 0 and not hasattr(self.alignment_basis, "shift"):
            raise RuntimeError(
                f"{self.__class__.__name__}'s alignment_basis {self.alignment_basis} must provide a `shift` method."
            )

        self.sub_pixel = sub_pixel

        # Configure number of radial points
        self.n_radial = n_radial or self.src.L // 2

        # Setup Polar Transform
        self._pft = PolarFT(
            self.src.L, ntheta=n_angles, nrad=self.n_radial, dtype=self.dtype
        )
        self._mask = xp.asarray(grid_2d(self.src.L, normalized=True)["r"] < 1)

    def _fast_rotational_alignment(self, pfA, pfB):
        """
        Perform fast rotational alignment using Polar Fourier cross correlation.

        Note broadcasting is specialized for this problem.
        pfA.shape (m, ntheta, nrad)
        pfB.shape (n, ntheta, nrad)
        yields thetas (m,n), peaks (m,n)

        """

        if pfA.ndim == 2:
            pfA = pfA[None]
        if pfB.ndim == 2:
            pfB = pfB[None]

        # 2 hats one sum
        pfA = fft.fft(pfA, axis=-2)
        pfB = fft.fft(pfB, axis=-2)
        # Tabulate elements of pfA cross pfB.conj() using broadcast multiply
        x = xp.expand_dims(pfA, 1) * xp.expand_dims(pfB.conj(), 0)
        angular = xp.sum(
            fft.ifft(x, axis=-2).real, axis=-1
        )  # sum all radial contributions

        # Resolve the angle maximizing the correlation through the angular dimension
        inds = xp.argmax(angular, axis=-1)

        max_thetas = 2 * np.pi / self._pft.ntheta * inds
        peaks = xp.take_along_axis(angular, inds[..., None], axis=-1).squeeze(-1)

        return xp.asnumpy(max_thetas), xp.asnumpy(peaks)

    def align(self, classes, reflections, basis_coefficients=None):
        """
        See `AligningAverager2D.align`
        """

        # Admit simple case of single case alignment
        classes = np.atleast_2d(classes)
        reflections = np.atleast_2d(reflections)

        # Result arrays
        # These arrays will incrementally store our best alignment.
        n_classes, n_nbor = classes.shape
        rotations = np.zeros((n_classes, n_nbor), dtype=self.dtype)
        dot_products = np.ones((n_classes, n_nbor), dtype=self.dtype) * -np.inf
        shifts = np.zeros((*classes.shape, 2), dtype=self.dtype)

        # Create a search grid and force initial pair to (0,0)
        # This is done primarily in case of a tie later, we would prefer unshifted.
        test_shifts = self._shift_search_grid(
            self.src.L,
            self.radius,
            roll_zero=True,
            sub_pixel=self.sub_pixel,
        )

        # Work arrays
        bs = min(self.batch_size, len(test_shifts))
        _rotations = np.zeros((bs, n_nbor), dtype=self.dtype)
        _dot_products = np.ones((bs, n_nbor), dtype=self.dtype) * -np.inf
        template_images = xp.empty(
            (bs, self._pft.ntheta // 2, self._pft.nrad), dtype=complex_type(self.dtype)
        )
        _images = xp.empty((n_nbor - 1, self.src.L, self.src.L), dtype=self.dtype)

        for k in trange(n_classes, desc="Rotationally aligning classes"):
            # We want to locally cache the original images,
            #  because we will mutate them with shifts in the next loop.
            #  This avoids recomputing them before each shift
            # The coefficient for the base images are also computed here.
            if basis_coefficients is None:
                original_images = Image(self._cls_images(classes[k], src=self.src))
            else:
                original_coef = basis_coefficients[classes[k], :]
                original_images = self.alignment_basis.evaluate(original_coef)

            _img0 = original_images[0].asnumpy().copy()
            _images[:] = xp.asarray(original_images[1:].asnumpy().copy())

            # Handle reflections
            refl = reflections[k][1:]  # skips original_image 0
            _images[refl] = xp.flip(_images[refl], axis=-2)

            # Mask off
            _images[:] = _images[:] * self._mask

            # Convert to polar Fourier
            pf_img0 = self._pft._transform(_img0)
            pf_images = self._pft.half_to_full(self._pft._transform(_images))

            # Batch over shift search space, updating best results
            pbar = tqdm(
                total=len(test_shifts),
                desc="\tmaximizing over shifts",
                disable=len(test_shifts) == 1,
                leave=False,
            )
            for start in range(0, len(test_shifts), self.batch_size):
                end = min(start + self.batch_size, len(test_shifts))
                bs = end - start  # handle a small last batch
                batch_shifts = test_shifts[start:end]

                # Shift the base, pf_img0, for each shift in this batch
                # Note this includes shifting for the zero shift case
                template_images[:bs] = xp.asarray(
                    self._pft.shift(pf_img0, batch_shifts)
                )

                pf_template_images = self._pft.half_to_full(template_images)

                # Compute and assign the best rotation found with this translation
                # note offset of 1 for skipped original_image 0
                _rotations[:bs, 1:], _dot_products[:bs, 1:] = (
                    self._fast_rotational_alignment(pf_template_images[:bs], pf_images)
                )

                # Note, these could be vectorized, but the code block
                # wasn't appreciably faster when I compared them for
                # current problem sizes.
                for i in range(bs):

                    # Test and update
                    # Each base-neighbor pair may have a best shift+rot from a different shift iteration.
                    improved_indices = _dot_products[i] > dot_products[k]
                    rotations[k, improved_indices] = -_rotations[i, improved_indices]
                    dot_products[k, improved_indices] = _dot_products[
                        i, improved_indices
                    ]
                    # base shifts assigned here, commutation resolved end of loop
                    shifts[k, improved_indices] = -batch_shifts[i]

                pbar.update(bs)

            # Completed batching over shifts
            pbar.close()

            # Commute the rotation and shift (code shifted the base image instead of all class members)
            shifts[k] = commute_shift_rot(shifts[k], rotations[k])

        return rotations, shifts, dot_products


class EMAverager2D(Averager2D):
    """
    Citation needed.
    """


class FTKAverager2D(Averager2D):
    """
    Factorization of the translation kernel for fast rigid image alignment.
    Rangan, A.V., Spivak, M., Anden, J., & Barnett, A.H. (2019).
    """
