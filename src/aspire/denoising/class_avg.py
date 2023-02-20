import logging

import numpy as np

from aspire.basis import FFBBasis2D
from aspire.classification import (
    Averager2D,
    BFRAverager2D,
    BFSReddyChatterjiAverager2D,
    Class2D,
    ClassSelector,
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
        classification_src,
        classifier,
        class_selector,
        averager,
        averager_src=None,
    ):
        """
        Constructor of an object for denoising 2D images using class averaging methods.

        :param classification_src: Source used for image classification.
        :param classifier: Class2D subclass used for image classification.
            Example, RIRClass2D.
        :param class_selector: A ClassSelector subclass.
        :param averager: An Averager2D subclass.
        :param averager_src: Optional, Source used for image registration and averaging.
            Defaults to `classification_src`.
        """
        self.classification_src = classification_src
        if not isinstance(self.classification_src, ImageSource):
            raise ValueError(
                f"`classification_src` should be subclass of `ImageSource`, found {self.classification_src}."
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
                f"`averager` should be instance of `Averger2D`, found {self.averager_src}."
            )

        self.averager_src = averager_src
        if self.averager_src is None:
            self.averager_src = self.classification_src
        if not isinstance(self.averager_src, ImageSource):
            raise ValueError(
                f"`averager_src` should be subclass of `ImageSource`, found {self.averager_src}."
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
            L=self.averager_src.L,
            n=self.averager_src.n,
            dtype=self.averager_src.dtype,
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
        self.selection_indices = self.class_selector.select(
            self._nn_classes,
            self._nn_reflections,
            self._nn_distances,
        )

        # Override the initial self.n
        # Some selectors will (dramitcally) reduce the space of classes.
        if len(self.selection_indices) != self.n:
            logger.info(
                f"After selection process, updating maximum {len(self.selection_indices)} classes from {self.n}."
            )
        self.n = len(self.selection_indices)

        self._selected = True

    def _images(self, indices):
        """
        Output images
        """

        # Lazy evaluate the class selection
        if not self._selected:
            self._class_select()

        # check for wholly cached image sets first
        if self._cached_im is not None:
            logger.debug(f"Loading {len(indices)} images from cache")
            im = Image(self._cached_im[indices, :, :])

        # # check for cached image sets from class_selector
        # elif hasattr(self.class_selector, "heap") and (
        #     heap_inds := set(indices).intesection(self.class_selector._heap_ids)
        # ):
        #     # logger.debug(f"Mapping {len(heap_inds)} images from heap cache.")

        #     # Images in heap_inds can be fetched from class_selector.
        #     # For others, create an indexing map that preserves original order.
        #     indices_to_compute = {
        #         ind: i for i, ind in enumerate(indices) if i not in heap_inds
        #     }
        #     indices_from_heap = {
        #         ind: i for i, ind in enumerate(indices) if i in heap_inds
        #     }

        #     # Recursion, heap_inds set should be empty in the recursive call.
        #     computed_imgs = self._images(list(indices_to_compute.values()))
        #     # Get the map once to avoid traversing heap in a loop
        #     heapd = self._heap_id_dict

        #     imgs = np.empty(
        #         (len(indices), computed_imgs.resolution, computed_imgs.resolution),
        #         dtype=computed_imgs.dtype,
        #     )

        #     _inds = list(indices_to_compute.values())
        #     imgs[_inds] = computed_imgs
        #     _inds = list(indices_from_heap.values())
        #     imgs[_inds] = map(
        #         lambda k: self.class_selector.heap[heapd[k]][2],
        #         list(indices_from_heap.keys()),
        #     )
        #     # for (i,ind) in enumerate(indices):
        #     #     # collect the images
        #     #     if ind in heapd:
        #     #         loc = heapd[ind]
        #     #         assert self.class_selector.heap[loc][1] == ind
        #     #         _img = self.class_selector.heap[loc][2]
        #     #     else:
        #     #         _img = computed_imgs[indices_to_compute[ind]]
        #     #     # assign the image
        #     #     imgs[i] = _img

        #     return imgs

        else:
            # Perform image averaging for the requested classes
            logger.debug(f"Averaging {len(indices)} images from source")
            im = self.averager.average(
                self._nn_classes[indices], self._nn_reflections[indices]
            )

        # Finally, apply transforms to resulting Image
        return self.generation_pipeline.forward(im, indices)


class DebugClassAvgSource(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    Packs base with common debug defaults.
    """

    def __init__(
        self,
        classification_src,
        n_nbor=10,
        num_procs=1,  # Change to "auto" if your machine has many processors
        classifier=None,
        class_selector=None,
        averager=None,
    ):
        dtype = classification_src.dtype

        if classifier is None:
            classifier = RIRClass2D(
                classification_src,
                fspca_components=400,
                bispectrum_components=300,  # Compressed Features after last PCA stage.
                n_nbor=n_nbor,
                large_pca_implementation="legacy",
                nn_implementation="legacy",
                bispectrum_implementation="legacy",
            )

        if averager is None:
            averager = BFRAverager2D(
                FFBBasis2D(classification_src.L, dtype=classification_src.dtype),
                classification_src,
                num_procs=num_procs,
                dtype=dtype,
            )

        if class_selector is None:
            class_selector = TopClassSelector()

        super().__init__(
            classification_src=classification_src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
            averager_src=classification_src,
        )


class LegacyClassAvgSource(ClassAvgSource):
    """
    Source for denoised 2D images using class average methods.

    Packs base with common v9 and v10 defaults.
    """

    def __init__(
        self,
        classification_src,
        n_nbor=10,
        num_procs=1,  # Change to "auto" if your machine has many processors
        classifier=None,
        class_selector=None,
        averager=None,
    ):
        dtype = classification_src.dtype

        if classifier is None:
            classifier = RIRClass2D(
                classification_src,
                fspca_components=400,
                bispectrum_components=300,  # Compressed Features after last PCA stage.
                n_nbor=n_nbor,
                large_pca_implementation="legacy",
                nn_implementation="sklearn",  # Note this is different than debug
                bispectrum_implementation="legacy",
            )

        if averager is None:
            basis_2d = FFBBasis2D(classification_src.L, dtype=dtype)
            averager = BFSReddyChatterjiAverager2D(
                basis_2d, classification_src, num_procs=num_procs, dtype=dtype
            )

        if class_selector is None:
            class_selector = TopClassSelector()

        super().__init__(
            classification_src=classification_src,
            classifier=classifier,
            class_selector=class_selector,
            averager=averager,
            averager_src=classification_src,
        )
