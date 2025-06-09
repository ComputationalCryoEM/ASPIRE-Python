import logging

import numpy as np

from aspire.basis import FFBBasis2D
from aspire.classification import (
    Averager2D,
    BandedSNRImageQualityFunction,
    BFRAverager2D,
    BFTAverager2D,
    Class2D,
    ClassSelector,
    GlobalVarianceClassSelector,
    GlobalWithRepulsionClassSelector,
    RIRClass2D,
    TopClassSelector,
)
from aspire.image import Image
from aspire.source import ImageSource

logger = logging.getLogger(__name__)


class ClassAvgSource(ImageSource):
    """
    Source for denoised 2D images using class average methods.
    """

    def __init__(
        self,
        src,
        classifier,
        class_selector,
        averager,
        batch_size=512,
    ):
        """
        Constructor of an object for denoising 2D images using class averaging methods.

        :param src: Source used for image classification.
        :param classifier: Class2D subclass used for image classification.
            Example, RIRClass2D.
        :param class_selector: A ClassSelector subclass.
        :param averager: An Averager2D subclass.
        :param batch_size: Integer size for batched operations.
        """
        self.src = src
        self.batch_size = int(batch_size)
        if not isinstance(self.src, ImageSource):
            raise ValueError(
                f"`src` should be subclass of `ImageSource`, found {self.src}."
            )

        self.classifier = classifier
        if not isinstance(self.classifier, Class2D):
            raise ValueError(
                f"`classifier` should be subclass of `Class2D`, found {self.classifier}."
            )

        self.class_selector = class_selector
        if not isinstance(self.class_selector, ClassSelector):
            raise ValueError(
                f"`class_selector` should be instance of `ClassSelector`, found {class_selector}."
            )

        self.averager = averager
        if not isinstance(self.averager, Averager2D):
            raise ValueError(
                f"`averager` should be instance of `Averager2D`, found {self.averager}."
            )

        # Flag for lazy eval, we'll classify once, on first touch.
        self._classified = False
        self._selected = False

        # Note n will potentially be updated after class selection.
        # Manage delayed setting `n` once.
        self._n_set = False

        super().__init__(
            L=self.averager.src.L,
            n=self.averager.src.n,
            dtype=self.averager.src.dtype,
            symmetry_group=self.src.symmetry_group,
            pixel_size=self.src.pixel_size,
        )

        # Any further operations should not mutate this instance.
        # Note, updating `n` is a special case for this source at this time.
        self._mutable = False

    @ImageSource.n.setter
    def n(self, n):
        """
        Sets max image index `n` in `src` and associated
        `ImageAccessor`.

        :param n: Number of images.
        """

        # Resets the protection of self._n exactly once after __init__.
        if (self._n is not None) and (not self._n_set):
            self._n = None
            self._n_set = True

        super()._set_n(n)

    def _classify(self):
        """
        Perform the image classification (if not already done).
        """

        # Short circuit
        if self._classified:
            logger.debug(f"{self.__class__.__name__} already classified, skipping")
            return

        (
            self.class_indices,
            self.class_refl,
            self.class_distances,
        ) = self.classifier.classify()
        self._classified = True

    @property
    def selection_indices(self):
        self._class_select()
        return super().selection_indices

    @selection_indices.setter
    def selection_indices(self, value):
        self.set_metadata(["_selection_indices"], value)

    @property
    def class_indices(self):
        """
        Returns table of class image indices as `(src.n, n_nbors)`
        Numpy array.

        Each row reprsents a class, with the columns ordered by
        smallest `class_distances` from the reference image (zeroth
        columm).

        Note `n_nbors` is managed by `self.classifier` and used here
        for documentation.

        :return: Numpy array, integers.
        """
        self._classify()
        return super().class_indices

    @class_indices.setter
    def class_indices(self, table):
        self.set_metadata(
            ["_class_indices"], [",".join(map(str, row)) for row in table]
        )

    @property
    def class_refl(self):
        """
        Returns table of class image reflections as `(src.n, n_nbors)`
        Numpy array.

        Follows same layout as `class_indices` but holds booleans that
        are True when the image should be reflected before averaging.

        Note `n_nbors` is managed by `self.classifier` and used here
        for documentation.

        :return: Numpy array, boolean.
        """
        self._classify()
        return super().class_refl

    @class_refl.setter
    def class_refl(self, table):
        # Convert boolean to (O, 1) integers.
        array_int = np.array(table, dtype=int)
        self.set_metadata(
            ["_class_refl"], [",".join(map(str, row)) for row in array_int]
        )

    @property
    def class_distances(self):
        """
        Returns table of class image distances as `(src.n, n_nbors)`
        Numpy array.

        Follows same layout as `class_indices` but holds floats
        representing the distance (returned by classifier) to the
        zeroth image in each class.

        Note `n_nbors` is managed by `self.classifier` and used here
        for documentation.

        :return: Numpy array, self.dtype.
        """
        self._classify()
        return super().class_distances

    @class_distances.setter
    def class_distances(self, table):
        self.set_metadata(
            ["_class_distances"], [",".join(map(str, row)) for row in table]
        )

    def _class_select(self):
        """
        Uses the `class_selector` in conjunction with the classifier results
        to select the classes (and order) used for image averager,
        if not already done.
        """

        # Short circuit
        if self._selected:
            logger.debug(
                f"{self.__class__.__name__} already selected classes, skipping"
            )
            return

        # Evaluate the classification network.
        if not self._classified:
            self._classify()

        # Perform class selection
        _selection_indices = self.class_selector.select(
            self.class_indices,
            self.class_refl,
            self.class_distances,
        )

        # Override the initial self.n
        # Some selectors will (dramatically) reduce the space of classes.
        if len(_selection_indices) != self.n:
            logger.info(
                f"After selection process, updating maximum {len(_selection_indices)} classes from {self.n}."
            )
        # Note, setter should be inherited from base ImageSource.
        self.n = len(_selection_indices)

        self._selected = True
        self.selection_indices = _selection_indices

    def save(self, *args, **kwargs):
        """
        Save metadata to STAR file.

        See `ImageSource.save` for documentation.
        """
        # Evaluate any lazy actions.
        # This should populate relevant metadata.
        self._class_select()
        # Call parent `save` method.
        return super().save(*args, **kwargs)

    def _images(self, indices):
        """
        Output images
        """

        # Lazy evaluate the class selection
        if not self._selected:
            self._class_select()

        # Truncate the request if nessecary,
        # ie, when selection reduces `self.n`.
        indices = np.array(indices, dtype=int)
        selected = indices[indices < self.n]

        if len(indices) != len(selected):
            deselected = indices[indices >= self.n]
            logger.debug(
                f"Dropping requested indices {deselected} following to selection process."
            )
        indices = selected

        # Remap to the selected ordering
        _indices = indices.copy()  # Store original mapping
        indices = self.selection_indices[indices]

        # Check if there is a cache available from class selection component.
        # Note, we can use := for this in the branch directly, when Python>=3.8
        heap_inds = None
        # Check we are using the same averager before attempting to use heap.
        if hasattr(self.class_selector, "heap") and (
            self.averager == self.class_selector.averager
        ):
            # Then check if request matches anything in the heap.
            heap_inds = set(indices).intersection(self.class_selector.heap_ids)

        # Check if this src cached images.
        if self._cached_im is not None:
            logger.debug(
                f"Loading {len(indices)} images from image cache, indices {_indices}"
            )
            im = self._cached_im[_indices, :, :]

        # Check for heap cached image sets from class_selector.
        elif heap_inds:
            logger.debug(
                f"Mapping {len(heap_inds)} images from heap cache, indices {indices}"
            )

            # Images in heap_inds can be fetched from class_selector.
            # For others, create an indexing map that preserves
            # original order.  Both of these dicts map requested image
            # id to location in the return array `im`.

            # Note, this stores inds from the remapped `indices`.
            indices_from_heap = {
                ind: i for i, ind in enumerate(indices) if ind in heap_inds
            }
            # Note, this stores _inds from the original `_indices`.
            # This allows the recusive call, which remaps the indices.
            indices_to_compute = {
                _ind: i
                for i, _ind in enumerate(_indices)
                if self.selection_indices[_ind] not in heap_inds
            }

            # Get heap dict once to avoid traversing heap in a loop.
            heap_dict = self.class_selector.heap_idx_map

            # Create an empty array to pack results.
            L = self.averager.src.L
            im = np.empty(
                (len(indices), L, L),
                dtype=self.averager.dtype,
            )

            # Recursively call `_images`.
            # `heap_inds` set should be empty in the recursive call,
            # and compute only remaining images (those not in heap).
            _compute_indices = list(indices_to_compute.keys())
            # Skip when empty (everything requested in heap).
            if len(_compute_indices):
                _imgs = self._images(_compute_indices)

                # Pack images computed from `_images` recursive call.
                _inds = list(indices_to_compute.values())
                im[_inds] = _imgs

            # Pack images from heap.
            for k, i in indices_from_heap.items():
                # map the image index to heap item location
                heap_loc = heap_dict[k]
                im[i] = self.class_selector.heap[heap_loc].image

            # Finally construct an Image.
            im = Image(im)

        else:
            # Perform image averaging for the requested images (classes)
            logger.debug(
                f"Averaging {len(indices)} images from source, indices: {indices}"
            )
            im = self.averager.average(
                self.class_indices[indices], self.class_refl[indices]
            )

        # Finally, apply transforms to resulting Images
        return self.generation_pipeline.forward(im, indices)

    def _get_classifier_basis(self, classifier):
        """
        Returns underlying basis of a classifier.

        For classifiers using compressed basis,
        returns the underlying uncompressed basis.

        Defaults to `FFBBasis2D` when `pca_basis` is not found.

        :param classifier: Class2D subclass to query.
        :return: `classifier` basis
        """

        if hasattr(classifier, "pca_basis") and classifier.pca_basis is not None:
            basis = classifier.pca_basis.basis
        else:
            # In the cases where a basis is not defined yet,
            #   construct a FFBBasis2D default.
            basis = FFBBasis2D(classifier.src.L, dtype=classifier.dtype)

        return basis


# The following sub classes attempt to pack sensible defaults
#   into ClassAvgSource so that users don't need to
#   instantiate every component to get started.


class DebugClassAvgSource(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    In this context Debug means defaulting to:
        * Using the defaults for `RIRClass2D`.
        * Using `TopClassSelector` to select all classes maintaining the same order as the input source.
        * Using `BFRAverager2D` with defaults on a single core.
    """

    def __init__(
        self,
        src,
        n_nbor=10,
        classifier=None,
        class_selector=None,
        averager=None,
        batch_size=512,
    ):
        """
        Instantiates with default debug paramaters.

        :param src: Source used for image classification.
        :param n_nbor: Number of nearest neighbors. Default 10.
        :param classifier: `Class2D` classifier instance.
            Default `None` creates `RIRClass2D`.
            See code for parameter details.
        :param class_selector: `ClassSelector` instance.
            Default `None` creates `TopClassSelector`.
        :param averager: `Averager2D` instance.
            Default `None` creates `BFRAverager2D` instance.
            See code for parameter details.
        :param batch_size: Integer size for batched operations.

        :return: ClassAvgSource instance.
        """
        dtype = src.dtype

        if classifier is None:
            classifier = RIRClass2D(
                src,
                fspca_components=400,
                bispectrum_components=300,  # Compressed Features after last PCA stage.
                n_nbor=n_nbor,
                large_pca_implementation="legacy",
                nn_implementation="legacy",
                bispectrum_implementation="legacy",
            )

        if averager is None:
            averager = BFRAverager2D(
                self._get_classifier_basis(classifier),
                src,
                dtype=dtype,
                batch_size=batch_size,
            )

        if class_selector is None:
            class_selector = TopClassSelector()

        super().__init__(
            src=src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
            batch_size=batch_size,
        )


class LegacyClassAvgSource(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    Defaults to using global variance based class selection, and a
    rotational image alignment.  Translational alignment is skipped by
    default (images are assumed reasonably centered), but can be
    configured by supplying a custom `averager=BFTAverager2D(...)`
    argument.

    This is similar to what was reported for papers using the
    MATLAB code.
    """

    def __init__(
        self,
        src,
        n_nbor=50,
        classifier=None,
        class_selector=None,
        averager=None,
        averager_src=None,
        batch_size=512,
    ):
        """
        Instantiates `ClassAvgSource` with the following parameters.

        :param src: Source used for image classification.
        :param n_nbor: Number of nearest neighbors. Default 50.
        :param classifier: `Class2D` classifier instance.
            Default `None` creates `RIRClass2D`.
            See code for parameter details.
        :param class_selector: `ClassSelector` instance.
            Default `None` creates `GlobalVarianceClassSelector`.
        :param averager: `Averager2D` instance.
            Default `None` creates `BFTAverager2D` instance.
            See code for parameter details.
        :param averager_src: Optionally explicitly assign source to
            `averager` during initialization.  Allows users to
             provide distinct sources for classification and
             averaging.  Raises error when combined with an explicit
             `averager` argument.
        :param batch_size: Integer size for batched operations.

        :return: ClassAvgSource instance.
        """
        dtype = src.dtype

        if classifier is None:
            classifier = RIRClass2D(
                src,
                fspca_components=400,
                bispectrum_components=300,  # Compressed Features after last PCA stage.
                n_nbor=n_nbor,
                large_pca_implementation="legacy",
                nn_implementation="sklearn",  # Note this is different than debug
                bispectrum_implementation="legacy",
            )

        if averager is None:
            if averager_src is None:
                averager_src = src

            basis_2d = self._get_classifier_basis(classifier)

            averager = BFTAverager2D(
                composite_basis=basis_2d,
                src=averager_src,
                radius=0,  # disables translation search
                batch_size=batch_size,
                dtype=dtype,
            )
        elif averager_src is not None:
            raise RuntimeError(
                "When providing an instantiated `averager`, cannot assign `averager_src`."
            )

        if class_selector is None:
            class_selector = GlobalVarianceClassSelector(averager=averager)

        super().__init__(
            src=src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
            batch_size=batch_size,
        )


def DefaultClassAvgSource(
    src,
    n_nbor=50,
    classifier=None,
    class_selector=None,
    averager=None,
    averager_src=None,
    batch_size=512,
    version=None,
):
    """
    Source for denoised 2D images using class average methods.

    Accepts `version`, to dispatch ClassAvgSource with parameters
    below.  Default `version` is latest available.  Different versions
    may have different defaults.

    :param src: Source used for image classification.
    :param n_nbor: Number of nearest neighbors. Default 50.
    :param classifier: `Class2D` classifier instance.
        Default `None` creates `RIRClass2D`.
        See code for parameter details.
    :param class_selector: `ClassSelector` instance.
    :param averager: `Averager2D` instance.
    :param averager_src: Optionally explicitly assign source to
        `averager` during initialization.  Allows users to
         provide distinct sources for classification and
         averaging.  Raises error when combined with an explicit
         `averager` argument.
    :param batch_size: Integer size for batched operations.
    :param version: Optionally selects a versioned `DefaultClassAvgSource`.
        Defaults to latest available.
    :return: ClassAvgSource instance.
    """

    _versions = {
        None: ClassAvgSourcev140,
        "latest": ClassAvgSourcev140,
        "0.14.0": ClassAvgSourcev140,
        "0.13.2": ClassAvgSourcev132,
    }

    if version not in _versions:
        raise RuntimeError(f"DefaultClassAvgSource version {version} not found.")
    cls = _versions[version]

    return cls(
        src,
        n_nbor=n_nbor,
        classifier=classifier,
        class_selector=class_selector,
        averager=averager,
        averager_src=averager_src,
        batch_size=batch_size,
    )


class ClassAvgSourcev140(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    Defaults to using global variance based class selection,
    and a brute force image alignment (rotational only).

    This is most similar to what was reported for papers using the
    MATLAB code, but takes significant time to compute.
    """

    def __init__(
        self,
        src,
        n_nbor=50,
        classifier=None,
        class_selector=None,
        averager=None,
        averager_src=None,
        batch_size=512,
    ):
        """
        Instantiates ClassAvgSourcev140 with the following parameters.

        :param src: Source used for image classification.
        :param n_nbor: Number of nearest neighbors. Default 50.
        :param classifier: `Class2D` classifier instance.
            Default `None` creates `RIRClass2D`.
            See code for parameter details.
        :param class_selector: `ClassSelector` instance.
            Default `None` creates `GlobalVarianceClassSelector`.
        :param averager: `Averager2D` instance.
            Default `None` creates `BFTAverager2D` instance.
            See code for parameter details.
        :param averager_src: Optionally explicitly assign source to
            `averager` during initialization.  Allows users to
             provide distinct sources for classification and
             averaging.  Raises error when combined with an explicit
             `averager` argument.
        :param batch_size: Integer size for batched operations.

        :return: ClassAvgSource instance.
        """
        dtype = src.dtype

        if classifier is None:
            classifier = RIRClass2D(
                src,
                fspca_components=400,
                bispectrum_components=300,  # Compressed Features after last PCA stage.
                n_nbor=n_nbor,
                large_pca_implementation="legacy",
                nn_implementation="sklearn",  # Note this is different than debug
                bispectrum_implementation="legacy",
            )

        if averager is None:
            if averager_src is None:
                averager_src = src

            basis_2d = self._get_classifier_basis(classifier)

            averager = BFTAverager2D(
                composite_basis=basis_2d,
                src=averager_src,
                batch_size=batch_size,
                dtype=dtype,
            )
        elif averager_src is not None:
            raise RuntimeError(
                "When providing an instantiated `averager`, cannot assign `averager_src`."
            )

        if class_selector is None:
            class_selector = GlobalVarianceClassSelector(averager=averager)

        super().__init__(
            src=src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
            batch_size=batch_size,
        )


class ClassAvgSourcev132(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    Defaults to using SNR based class selection,
    avoiding neighbors of previous classes,
    and a brute force image alignment (rotational only).
    """

    def __init__(
        self,
        src,
        n_nbor=50,
        classifier=None,
        class_selector=None,
        averager=None,
        averager_src=None,
        batch_size=512,
    ):
        """
        Instantiates ClassAvgSourcev132 with the following parameters.

        :param src: Source used for image classification.
        :param n_nbor: Number of nearest neighbors. Default 50.
        :param classifier: `Class2D` classifier instance.
            Default `None` creates `RIRClass2D`.
            See code for parameter details.
        :param class_selector: `ClassSelector` instance.
            Default `None` creates `GlobalWithRepulsionClassSelector` with
            `BandedSNRImageQualityFunction`. This will select the
            images with the highest banded SNR.
        :param averager: `Averager2D` instance.
            Default `None` creates `BFRAverager2D` instance.
            See code for parameter details.
        :param averager_src: Optionally explicitly assign source to
            `averager` during initialization.  Allows users to
             provide distinct sources for classification and
             averaging.  Raises error when combined with an explicit
             `averager` argument.
        :param batch_size: Integer size for batched operations.

        :return: ClassAvgSource instance.
        """
        dtype = src.dtype

        if classifier is None:
            classifier = RIRClass2D(
                src,
                fspca_components=400,
                bispectrum_components=300,  # Compressed Features after last PCA stage.
                n_nbor=n_nbor,
                large_pca_implementation="legacy",
                nn_implementation="sklearn",  # Note this is different than debug
                bispectrum_implementation="legacy",
            )

        if averager is None:
            if averager_src is None:
                averager_src = src

            basis_2d = self._get_classifier_basis(classifier)

            averager = BFRAverager2D(
                composite_basis=basis_2d,
                src=averager_src,
                batch_size=batch_size,
                dtype=dtype,
            )
        elif averager_src is not None:
            raise RuntimeError(
                "When providing an instantiated `averager`, cannot assign `averager_src`."
            )

        if class_selector is None:
            quality_function = BandedSNRImageQualityFunction()
            class_selector = GlobalWithRepulsionClassSelector(
                averager, quality_function
            )

        super().__init__(
            src=src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
            batch_size=batch_size,
        )
