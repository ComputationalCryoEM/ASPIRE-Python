import logging

from aspire.classification import Averager2D, Class2D, ClassSelector
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
            self.classification_src.n,
            self._nn_classes,
            self._nn_reflections,
            self._nn_distances,
        )

        # Override the initial self.n
        logger.info(f"Selecting {len(self.selection_indices)} of {self.n} classes.")
        self.n = len(self.selection_indices)

        self._selected = True

    def _images(self, indices):
        """
        Output images
        """

        # Lazy evaluate the class selection
        if not self._selected:
            self._class_select()

        # check for cached images first
        if self._cached_im is not None:
            logger.debug("Loading images from cache")
            im = Image(self._cached_im[indices, :, :])
        else:
            # Perform image averaging for the requested classes
            im = self.averager.average(
                self._nn_classes[indices], self._nn_reflections[indices]
            )

        # Finally, apply transforms to resulting Image
        return self.generation_pipeline.forward(im, indices)


# # Legacy
# :param num_procs: Number of processes to use.
#     `None` will attempt computing a suggestion based on machine resources.
#         # Setup class selection
#         if selector is None:
#             selector = RandomClassSelector(seed=self.seed)
#         elif not isinstance(selector, ClassSelector):
#             raise RuntimeError("`selector` must be subclass of `ClassSelector`")
#         self.selector = selector

#         self.averager = averager
#         # When not provided by a user, the averager is instantiated after
#         #  we are certain our pca_basis has been constructed.
#         if self.averager is None:
#             self.averager = BFSReddyChatterjiAverager2D(
#                 self.fb_basis, self.src, num_procs=self.num_procs, dtype=self.dtype
#             )
#         else:
#             # When user provides `averager` and `num_procs`
#             #   we should warn when `num_procs` mismatched.
#             if self.num_procs is not None and self.averager.num_procs != self.num_procs:
#                 logger.warning(
#                     f"{self.__class__.__name__} intialized with num_procs={self.num_procs} does not"
#                     f" match provided {self.averager.__class__.__name__}.{self.averager.num_procs}"
#                 )
