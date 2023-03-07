import logging

import numpy as np

from aspire.basis import FFBBasis2D
from aspire.classification import (
    Averager2D,
    BFRAverager2D,
    BFSRAverager2D,
    Class2D,
    ClassSelector,
    ContrastWithRepulsionClassSelector,
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
    ):
        """
        Constructor of an object for denoising 2D images using class averaging methods.

        :param src: Source used for image classification.
        :param classifier: Class2D subclass used for image classification.
            Example, RIRClass2D.
        :param class_selector: A ClassSelector subclass.
        :param averager: An Averager2D subclass.
        """
        self.src = src
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
                f"`averager` should be instance of `Averger2D`, found {self.averager}."
            )

        self._nn_classes = None
        self._nn_reflections = None
        self._nn_distances = None

        # Flag for lazy eval, we'll classify once, on first touch.
        # We could use self._nn_* vars, but might lose some flexibility later.
        self._classified = False
        self._selected = False

        # Note n will potentially be updated after class selection.
        super().__init__(
            L=self.averager.src.L,
            n=self.averager.src.n,
            dtype=self.averager.src.dtype,
        )

    def _classify(self):
        """
        Perform the image classification (if not already done).
        """

        # Short circuit
        if self._classified:
            logger.debug(f"{self.__class__.__name__} already classified, skipping")
            return

        (
            self._nn_classes,
            self._nn_reflections,
            self._nn_distances,
        ) = self.classifier.classify()
        self._classified = True

    @property
    def selection_indices(self):
        self._class_select()
        return self._selection_indices

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
        self._selection_indices = self.class_selector.select(
            self._nn_classes,
            self._nn_reflections,
            self._nn_distances,
        )

        # Override the initial self.n
        # Some selectors will (dramitcally) reduce the space of classes.
        if len(self._selection_indices) != self.n:
            logger.info(
                f"After selection process, updating maximum {len(self._selection_indices)} classes from {self.n}."
            )
        self._set_n(len(self._selection_indices))

        self._selected = True

    def _images(self, indices):
        """
        Output images
        """

        # Lazy evaluate the class selection
        if not self._selected:
            self._class_select()

        # Remap to the selected ordering
        indices = self.selection_indices[indices]

        # Check if there is a cache available from class selection component.
        # Note, we can use := for this in the branch directly, when Python>=3.8
        heap_inds = None
        if hasattr(self.class_selector, "heap"):
            # Then check if request matches anything in the heap.
            heap_inds = set(indices).intersection(self.class_selector.heap_ids)

        # Check if this src cached images.
        if self._cached_im is not None:
            logger.debug(f"Loading {len(indices)} images from image cache")
            im = Image(self._cached_im[indices, :, :])

        # Check for heap cached image sets from class_selector.
        elif heap_inds:
            logger.debug(f"Mapping {len(heap_inds)} images from heap cache.")

            # Images in heap_inds can be fetched from class_selector.
            # For others, create an indexing map that preserves
            # original order.  Both of these dicts map requested image
            # id to location in the return array `im`.
            indices_to_compute = {
                ind: i for i, ind in enumerate(indices) if i not in heap_inds
            }
            indices_from_heap = {
                ind: i for i, ind in enumerate(indices) if i in heap_inds
            }

            # Get heap dict once to avoid traversing heap in a loop.
            heapd = self.self.class_selector.heap_id_dict

            # Recursively call `_images`.
            # `heap_inds` set should be empty in the recursive call,
            # and compute only remaining images (those not in heap).
            _imgs = self._images(list(indices_to_compute.keys()))

            # Create an empty array to pack results.
            im = np.empty(
                (len(indices), _imgs.resolution, _imgs.resolution),
                dtype=_imgs.dtype,
            )

            # Pack images computed from `_images` recursive call.
            _inds = list(indices_to_compute.values())
            im[_inds] = _imgs

            # Pack images from heap.
            for k, i in indices_from_heap.items():
                # map the image index to heap item location
                heap_loc = heapd[k]
                im[i] = self.class_selector.heap[heap_loc].image

        else:
            # Perform image averaging for the requested images (classes)
            logger.debug(f"Averaging {len(indices)} images from source")
            im = self.averager.average(
                self._nn_classes[indices], self._nn_reflections[indices]
            )

        # Finally, apply transforms to resulting Images
        return self.generation_pipeline.forward(im, indices)


# The following sub classes attempt to pack sensible defaults
#   into ClassAvgSource so that users don't need to
#   instantiate every component to get started.


class DebugClassAvgSource(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    Defaults to `RIRClass2D`, `TopClassSelector`, `BFRAverager2D`
    using a single processor.
    """

    def __init__(
        self,
        src,
        n_nbor=10,
        num_procs=1,  # Change to "auto" if your machine has many processors
        classifier=None,
        class_selector=None,
        averager=None,
    ):
        """
        Instantiates with default debug paramaters.

        :param src: Source used for image classification.
        :param n_nbor: Number of nearest neighbors. Default 10.
        :param num_procs: Number of processors. Default of 1 runs serially.
            `None` attempts to compute a reasonable value
            based on available cores and memory.
        :param classifier: `Class2D` classifier instance.
            Default `None` creates `RIRClass2D`.
            See code for parameter details.
        :param class_selector: `ClassSelector` instance.
            Default `None` creates `TopClassSelector`.
        :param averager: `Averager2D` instance.
            Default `None` ceates `BFRAverager2D` instance.
            See code for parameter details.

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
                FFBBasis2D(src.L, dtype=src.dtype),
                src,
                num_procs=num_procs,
                dtype=dtype,
            )

        if class_selector is None:
            class_selector = TopClassSelector()

        super().__init__(
            src=src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
        )


class ClassAvgSourcev11(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    Defaults to using Contrast based class selection (on the fly, compressed),
    avoiding neighbors of previous classes,
    and a brute force image alignment.

    Currently this is the most reasonable default for experimental data.
    """

    def __init__(
        self,
        src,
        n_nbor=50,
        num_procs=None,
        classifier=None,
        class_selector=None,
        averager=None,
        averager_src=None,
    ):
        """
        Instantiates ClassAvgSourcev11 with the following parameters.

        :param src: Source used for image classification.
        :param n_nbor: Number of nearest neighbors. Default 50.
        :param num_procs: Number of processors. Use 1 to run serially.
            Default `None` attempts to compute a reasonable value
            based on available cores and memory.
        :param classifier: `Class2D` classifier instance.
            Default `None` creates `RIRClass2D`.
            See code for parameter details.
        :param class_selector: `ClassSelector` instance.
            Default `None` creates `ContrastWithRepulsionClassSelector`.
        :param averager: `Averager2D` instance.
            Default `None` ceates `BFSRAverager2D` instance.
            See code for parameter details.
        :param averager_src: Optionally explicitly assign source
            to BFSRAverager2D during initialization.
            Raises error when combined with an explicit `averager`
            argument.

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

            basis_2d = FFBBasis2D(averager_src.L, dtype=dtype)

            averager = BFSRAverager2D(
                composite_basis=basis_2d,
                src=averager_src,
                num_procs=num_procs,
                dtype=dtype,
            )
        elif averager_src is not None:
            raise RuntimeError(
                "When providing an instantiated `averager`, cannot assign `averager_src`."
            )

        if class_selector is None:
            class_selector = ContrastWithRepulsionClassSelector()

        super().__init__(
            src=src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
        )
