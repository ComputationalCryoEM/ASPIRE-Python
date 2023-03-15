Class Averaging Architecture
============================

ASPIRE now contains a broad collection of configurable and extensible
components which can be combined to create class averaging solutions
tailored to different datasets.  The architecture was designed to both
be modular and encourage experimentation.  Lower level components are
aggregated into a high level interface by ``ClassAvgSource``
instances.  Starting there this document will descend into each
contributing component.


ClassAvgSource
**************

``ClassAvgSource`` is the fully customizable base class which links
together components into a cohesive source to be used with other
ASPIRE components.  A power user can instantiate an instance of each
required component and assign them here for complete control.

.. mermaid::

   classDiagram
       class ClassAvgSource{
	   src: ImageSource
	   classifier: Class2D
	   class_selector: ClassSelector
	   averager: Averager2D
	   +images()
	   }

       ClassAvgSource o-- ImageSource
       ClassAvgSource o-- Class2D
       ClassAvgSource o-- ClassSelector
       ClassAvgSource o-- Averager2D

       class ImageSource{
	   +images()
       }
       class Class2D{
	   +classify()
       }
       class ClassSelector{
	   +select()
       }
       class Averager2D{
	   +average()
       }

""""""""""

While that allows for full customization, two helper classes are
provided that supply defaults as a jumping off point.  Both of these
helper sources only require an input ``Source`` to be instantiated.
They can still be fully customized, but they are intended to start
with sensible defaults, so users only need to instantiate the specific
components they wish to configure.

.. mermaid::

   classDiagram
      ClassAvgSource <|-- DebugClassAvgSource
      ClassAvgSource <|-- DefaultClassAvgSource
      class DebugClassAvgSource{
	 src: ImageSource
	 classifier: RIRClass2D
	 class_selector: TopClassSelector
	 averager: BFRAverager2D
	 +images()
      }
      class DefaultClassAvgSource{
	 version="11.0"
	 src: ImageSource
	 classifier: RIRClass2D
	 class_selector: ContrastWithRepulsionClassSelector
	 averager: BFSRAverager2D
	 +images()
      }

``DebugClassAvgSource`` is designed for use in testing, documentation,
and development because it defaults to the simplest components while
also maintaining the original input source index ordering.  That is,
the first 10 class averages from ``DebugClassAvgSource`` should
correspond with the first 10 source images without requiring any index
mappings etc.

``DefaultClassAvgSource`` applies the most sensible defaults available
in the current ASPIRE release.  ``DefaultClassAvgSource`` takes a
version string, such as ``11.0`` which will return a specific
configuration.  This version should allow users to perform a similar
experiment across releases as ASPIRE implements improved methods.
When a version is not provided, ``DefaultClassAvgSource`` defaults to
the latest version available.


Classifiers
***********

Classifiers take an image ``Source`` and attempts to classify into
``classes`` that identify images with similar viewing angles up to
reflection.  All ``Class2D`` instances are expected to implement a
``classify`` method which returns ``(classes, reflections,
distances)``.  The three returned variables are expected to be 2D
Numpy arrays in a neighbor network format having shape ``(n_classes,
n_nbors)``.  So to retrieve the input source indices for the first
class's neighbors, we would want ``classes[0,:]``.  The first class
index should be the base image used for classification.  So we would
expect ``classes[0,0]`` to be ``input_src.images[0]``.

No further class selection or order occurs during classification.
Those methods are broken out into other components.

Currently ASPIRE has a single classification algorithm known as
``RIRClass2D``.  This algorithm uses multiple applications of PCA in
conjunction with Bispectrum analysis to identify nearest neighbors in
a rotationally invariant feature space.

.. mermaid::

   classDiagram
      class Class2D{
	  +classify()
      }
    Class2D <|-- RIRClass2D

Class Selectors
***************

Class Selectors consume the output of ``Class2D`` and attempt to order
and/or filter classes down to a selection.  Selecting the "best"
classes in CryoEM problems is still an area of active research.  Some
common methods are provided, along with an extensible base interface.

Generally, Class Selection comes in two flavors depending on what
information is required to perform the selection.

Local Class Selectors
---------------------

For "Local" class selection, we will attempt to use only the
information returned from ``Class2D``.  In the case of ``RIRClass2D``
this would primarily be a network of ``distances`` as measured in the
compressed feature space.

This approach has two main advantages.  First, we already have this
information computed as part of classification.  Second, it allows us
to register and stack a relatively small subset of the "best" classes.
Because registration and alignment are computationally expensive this
can reduce pipeline run times by an order of magnitude.

.. mermaid::

   classDiagram
      class ClassSelector{
	 +select()
	 }
       ClassSelector <|-- TopClassSelector
       ClassSelector <|-- RandomClassSelector
       ClassSelector <|-- NeighborVarianceClassSelector
       ClassSelector <|-- DistanceClassSelector
       ClassSelector o-- GreedyClassRepulsion

Global Class Selectors
----------------------

Global Class Selection techniques first compute the entire collection
registered and aligned class averages, then compute some quality
measure on all classes.

These are most similar to the historical MATLAB approaches, sometimes
referred to as "out-of-core" methods.  It is believed that many legacy
MATLAB experiments computed contrast (variance) of all class averaged
images, sorted the class averages to express highest contrast,
potentially avoiding classes with views we've already seen.  This can
be accomplished now by using the ``ContrastImageQualityFunction`` in a
``GlobalWithRepulsionClassSelector``.

An SNR based approach is also provided, and a Bandpass method should
be implemented in a future release.  Again, these components are fully
customizable and the base interfaces were designed with algorithm
developers in mind.

Implementing concrete ``GlobalClassSelector`` leverage subcomponents
described below.

.. mermaid::

   classDiagram
       ClassSelector <|-- GlobalClassSelector
       class GlobalClassSelector{
	   averager: Averager2D
	   function: ImageQuaityFunction
	   heap_size: int
	   }
       GlobalClassSelector *-- ImageQualityFunction
       GlobalClassSelector ..> Heap
       GlobalClassSelector <|-- GlobalWithRepulsionClassSelector


       class ImageQualityFunction{
	  -_function
	  +__call__()
	  }
       ImageQualityFunction o-- WeightedImageQualityMixin
       ImageQualityFunction <|-- BandedSNRImageQualityFunction
       ImageQualityFunction <|-- ContrastImageQualityFunction
       ImageQualityFunction <|-- BandpassImageQualityFunction_TBD

       class WeightedImageQualityMixin{
	   -_weight_function
       }
       WeightedImageQualityMixin <|-- RampWeightedImageQualityMixin
       WeightedImageQualityMixin <|-- BumpWeightedImageQualityMixin

       GlobalClassSelector <|-- RampWeightedContrastImageQualityFunction
       RampWeightedImageQualityMixin <|-- RampWeightedContrastImageQualityFunction
       GlobalClassSelector <|-- BumpWeightedContrastImageQualityFunction
       BumpWeightedImageQualityMixin <|-- BumpWeightedContrastImageQualityFunction

Class Repulsion
^^^^^^^^^^^^^^^

Class Repulsion are techniques used to avoid classes based on some
criterion.  Currently we provide ``GreedyClassRepulsion``, but this
mix-in class can be mimicked to implement alternate schemes.

``GreedyClassRepulsion`` is based on the following intuition. Assume
the selection has in fact ordered the classes s.t. *the "best"
classes occur first*. It follows that the "best" expression of a
viewing angle locus will be the first seen.  Now assume *the
classifier returns classes with closest viewing angles* (up to
reflections).  Then the classes formed by *neighbors of the current
expression are inferior*.  The aggressiveness of the neighbor
repulsion count is tunable.

In practice, ``GreedyClassRepulsion`` is a mix-in designed to be mixed
into any other ``ClassSelector``.  Note, that repulsion can (and will)
dramatically reduce the population of class averages returned.


Image Quality Functions
^^^^^^^^^^^^^^^^^^^^^^^

The ``ImageQualityFunction`` interface provides a consistent way to
bring your own function to measure the quality of an aligned and
registered class average.  This function should operate on a single
Image, with conversions and broadcasting being handled behind the
scenes.

An example would be ``ContrastImageQualityFunction`` which computes
and returns contrast as variance.

Another advantage of using the class is that it exposes and manages a
grid cache, which is handy to avoid recomputing the same grid for
every image when using spatial methods.

WeightedImageQualityMixin
^^^^^^^^^^^^^^^^^^^^^^^^^

``WeightedImageQualityMixin`` is designed to mix with subclasses of
``ImageQualityFunction``, extending them with a weighted image mask
applied prior to the image quality function call.

Two concrete examples are provided
``BumpWeightedContrastImageQualityFunction`` and
``RampWeightedContrastImageQualityFunction`` which apply the
respective weight functions prior to the Contrast calculation.

Again, ``WeightedImageQualityMixin`` exposes and manages a grid cache,
this time for grid weights.


Averagers
*********

Averagers consume from a ``Source`` and return averaged images
defined by class network arguments ``classes`` and ``reflections``.
You may find the terms averaging and stacking used interchangeably in
this context, so know that averaging does not always imply *arithmetic
mean*,

Some averaging techniques, those subclassing ``AligningAverager2D``
have distinct ``alignment`` and ``averaging`` stages.  Others such as
Expectation-Maximization may perform these internally and provide only
an opaque ``averages`` stage.

.. mermaid::

   classDiagram
	class Averager2D{
	    basis: Basis
	    src: ImageSource
	    +average()
	}
	Averager2D ..> ImageStacker
	Averager2D <|-- AligningAverager2D
	class AligningAverager2D{
	    align()
	}
	ImageSource *-- Averager2D
	Averager2D <|-- AligningAverager2D
	Averager2D <|-- EMAverager2D_TBD
	Averager2D <|-- FTKAverager2D_TBD
	AligningAverager2D <|-- BFRAverager2D
	BFRAverager2D <|-- BFSRAverager2D
	AligningAverager2D <|-- ReddyChetterjiAverager2D
	ReddyChetterjiAverager2D <|-- BFSReddyChetterjiAverager2D

	class ImageStacker{
	    stack()
	}
	class SigmaRejectionImageStacker{
	    sigma
	}
	class WinsorizedImageStacker{
	    percentile
	}
	ImageStacker <|-- MeanImageStacker
	ImageStacker <|-- MedianImageStacker
	ImageStacker <|-- SigmaRejectionImageStacker
	SigmaRejectionImageStacker .. Gaussian
	SigmaRejectionImageStacker .. FWHM
	ImageStacker <|-- WinsorizedImageStacker

ImageStacker
------------

``ImageStacker`` provides an interface for the common task of stacking
images.  Implementations for common stacking methods are provided and
should work for both ``Image`` and (1D) coefficient stacks.  Users
experimenting with advanced stacking are responsible for selecting an
ImageStacker method appropriate for their data.

Note that the ASPIRE default is naturally ``MeanImageStacker``.

