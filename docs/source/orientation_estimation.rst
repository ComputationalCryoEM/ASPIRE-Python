Orientation Estimation Architecture
===================================

ASPIRE contains a collection of common-line algorithms for orientation estimation that
are tailored to datasets with various characteristics, such as particle symmetry or
viewing direction distribution. The solvers share infrastructure for polar Fourier
preparation, rotation and shift estimation, and GPU-aware common-line computation,
which makes it easy to swap algorithms within a pipeline or add new ones when novel data
characteristics demand it.

High-Level Workflow
-------------------

The reference-free workflow below illustrates how orientation solvers integrate with
denoising and reconstruction utilities.
     
#. Starting from an ``ImageSource`` (often the output of the class averaging stack
   described in :doc:`class_source`), instantiate one of the ``CLOrient3D`` subclasses
   with the source plus any tuning parameters (angular resolution, shift search ranges,
   histogram settings).

   .. code-block::
      
       # Instantiate orientation estimation object
       from aspire.abinitio import CLSync3N
       orient_est = CLSync3N(src, n_theta=180, shift_step=0.5)

#. Rotations and shifts can be estimated directly from the orientation solver by
   calling the methods ``estimate_rotations()`` and ``estimate_shifts()`` or by
   requesting them as attributes of the class (see below), which will initiate
   estimation if they do not already exist.

   .. code-block::
      
       est_rots = orient_est.rotations
       est_shifts = orient_est.shifts
      
#. Or, in the context of a full reconstruction pipeline, the image source and orientation
   estimation objects can be used to instantiate an ``OrientedSource`` to be consumed
   as input to a downstream volume reconstruction method. In this case, rotations and
   shifts will be estimated in a lazy fashion when requested by the reconstruction method.

   .. code-block::
      
       # Create an 'OrientedSource' to pass to a mean volume estimator
       from aspire.reconstruction import MeanEstimator
       from aspire.source import OrientedSource

       oriented_src = OrientedSource(src, orient_est)
       estimator = MeanEstimator(oriented_src)

       # Estimate volume
       est_vol = estimator.estimate()

       # Estimated rotations/shifts can be accessed via the 'OrientedSource'
       est_rots = orient_src.rotations
       est_shifts = orient_src.shifts

Layout of the Class Hierarchy
-----------------------------

All common-line estimators live under :mod:`aspire.abinitio` and share the base class
``CLOrient3D``. Algorithms that rely on a pairwise common-line matrix inherit from the
intermediary base class ``CLMatrixOrient3D``. Together they codify the data preparation
steps, caching strategy, and the minimal interface each subclass must expose.

CLOrient3D
^^^^^^^^^^

``CLOrient3D`` manages dataset-wide configuration such as polar grid resolution, masking
strategies, and the shift solving backend. It exposes:

- ``src``: the ``ImageSource`` supplying masked projection stacks.
- ``pf``: a lazily-evaluated :class:`aspire.operators.PolarFT` representation of the
  masked images produced by ``_prepare_pf``. The helper applies the ``fuzzy_mask`` with
  ``risetime=2`` before stripping the DC component, ensuring historical parity with the
  MATLAB pipeline.
- ``estimate_rotations()``: abstract method overridden by subclasses with the algorithm-
  specific synchronization logic.
- ``estimate_shifts()``: provided implementation that solves the sparse 2D shift system
  once global rotations are estimated.

.. mermaid::

   classDiagram
       class CLOrient3D{
           src: ImageSource
	   +estimate_rotations()
	   +estimate_shifts()
	   +pf
       }

CLMatrixOrient3D
^^^^^^^^^^^^^^^^

``CLMatrixOrient3D`` augments ``CLOrient3D`` with utilities for assembling the
``clmatrix`` (indices of correlated polar rays between image pairs) and any auxiliary
scores such as ``cl_dist`` or ``shifts_1d``. Key behaviors include:

- CPU/GPU dispatch within ``build_clmatrix``. When GPUs are available the class invokes
  CUDA kernels that drastically reduce wall time for large datasets.
- Caching and lazy-evaluation of ``clmatrix`` and distance matrices to avoid recomputation.
- Shared ``max_shift`` and ``shift_step`` parameters that influence accuracy/runtime
  trade-offs during 1D shift searches.

.. mermaid::

   classDiagram
       class CLMatrixOrient3D{
           src: ImageSource
	   +estimate_rotations()
	   +estimate_shifts()
	   +pf
	   +clmatrix
       }

Handedness Synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Many of the common-line algorithms involve solving for a set of relative rotations between
pairs of image orientations. Due to the inherent handedness ambiguity in the cryo-EM problem,
each estimated relative rotation might contain a spurious reflection. These spurious reflections
must be resolved such that all relative rotation estimates either contain a global reflection
or don't before downstream reconstruction. The :mod:`aspire.abinitio.J_sync` module implements a
power-method based handedness synchronization to complete this task.

- ``JSync`` builds a signed graph over triplets of relative rotations and iteratively
  estimates the optimal set of reflections that maximizes consistency of the recovered
  estimates.
- The solver supports CPU/GPU execution, configurable tolerances (``epsilon``) and
  iteration limits (``max_iters``), and logs residuals so that callers can monitor
  convergence.
- ``CLOrient3D`` subclasses simply import the ``JSync`` module to access handedness
  synchronization methods.

Class Hierarchy Overview
^^^^^^^^^^^^^^^^^^^^^^^^

The relationships below show how the concrete solvers inherit shared behaviors (masking, CL matrix
utilities, synchronization helpers) before layering on algorithm-specific logic.

.. mermaid::

   classDiagram
       CLOrient3D <|-- CLMatrixOrient3D
       CLMatrixOrient3D <|-- CLSync3N
       CLMatrixOrient3D <|-- CLSyncVoting   
       CLMatrixOrient3D <|-- CommonlineSDP
       CommonlineSDP <|-- CommonlineLUD
       CommonlineLUD <|-- CommonlineIRLS
       CLMatrixOrient3D <|-- CLSymmetryC2
       CLMatrixOrient3D <|-- CLSymmetryC3C4
       CLOrient3D <|-- CLSymmetryCn
       CLOrient3D <|-- CLSymmetryD2
       class JSync
       CLSync3N o-- JSync
       CLSymmetryC2 o-- JSync
       CLSymmetryC3C4 o-- JSync
       CLSymmetryCn o-- JSync  

Algorithms for Asymmetric Molecules
-----------------------------------

ASPIRE offers several orientation estimation algorithms for handling molecules with asymmetric data:

- ``CLSync3N`` (:file:`src/aspire/abinitio/commonline_sync3n.py`): ``CLSync3N`` detects
  common-lines between pairs of images and reduces misidentifications using a vote which
  incorporates information from all possible third images. This voting stage produces a
  set of pairwise rotations which are subsequently synchronized for handedness via ``Jsync``.
  These pairwise rotations are then used to form a 3Nx3N synchronization matrix
  which is optionally weighted to favor more statistically indicative pairwise rotations.
  An eigen-decomposition is then performed to simultaneously recover all image orientations.
- ``CLSyncVoting`` (:file:`src/aspire/abinitio/commonline_sync.py`): In ``CLSyncVoting``,
  the voting scheme directly populates a 2N×2N synchronization matrix of XY rotation blocks
  which is insensitive to the cryo-EM handedness ambiguity. For that reason a separate handedness
  synchronization step is not needed. An eigen-decomposition is then performed to recover the image
  orientations.
- ``CommonlineSDP`` (:file:`src/aspire/abinitio/commonline_sdp.py`): This method uses a
  relaxation of the least squares formulation of the orientation problem and frames it
  semidefinite program. ``cvxpy`` is used to solve the SDP and the rotations are recovered
  through deterministic rounding and ``nearest_rotations`` projection.
- ``CommonlineLUD`` (:file:`src/aspire/abinitio/commonline_lud.py`): Reuses the SDP
  scaffolding but substitutes an ADMM-based least unsquared deviations solver. Parameters
  like ``alpha``, ``mu`` scheduling, and adaptive rank selection govern convergence.
- ``CommonlineIRLS`` (:file:`src/aspire/abinitio/commonline_irls.py`): Wraps LUD inside
  an outer reweighting loop, updating residual weights and ``self._mu`` to improve
  robustness to outliers.

Algorithms for Symmetric Molecules
----------------------------------

Symmetry-aware variants search for multiple common lines per image pair and embed symmetry
group constraints while estimating rotations:

- ``CLSymmetryC2`` (:file:`src/aspire/abinitio/commonline_c2.py`): Estimates orientations
  for molecules with 2-fold cyclic symmetry by searching for two common-lines per image
  pair, construction a set of pairwise rotations, performing local and global handedness
  synchronization, and finally recovering the orientations from the synchronized relative
  rotations.
- ``CLSymmetryC3C4`` (:file:`src/aspire/abinitio/commonline_c3_c4.py`): Targets order-3 and
  order-4 cyclic molecules by detecting self-common-lines, forming relative third-row outer
  products, running a local/global ``JSync`` pass, then extracts two rows of each rotation
  matrix from the outer products and finally estimates in-plane rotations to recover full rotations.
- ``CLSymmetryCn`` (:file:`src/aspire/abinitio/commonline_cn.py`): Handles higher-order cyclic
  symmetry (n > 4) by generating a discretized set of candidate rotations on the sphere, evaluating
  likelihoods of induced common/self-common lines, pruning equatorial degeneracies, and synchronizing
  the surviving outer-product blocks before estimating in-plane angles.
- ``CLSymmetryD2`` (:file:`src/aspire/abinitio/commonline_d2.py`): Deals with dihedral symmetry
  via a Saff–Kuijlaars sphere grid, equator/top-view filtering, exhaustive lookup tables for
  relative rotations, and a color/sign synchronization stage that enforces the ``DnSymmetryGroup``
  constraints prior to assigning final rotations.

These classes reuse the ``estimate_shifts`` implementation once symmetry-consistent rotations are available.

Extensibility
-------------

Adding a new orientation estimator typically involves:

#. Subclassing ``CLOrient3D`` (or ``CLMatrixOrient3D`` if you need common-line matrices).
#. Implementing ``estimate_rotations(self, **kwargs)``. This method must return an
   ``(n, 3, 3)`` array of rotations to be consumed by ``estimate_shifts`` and downstream
   reconstruction methods.
#. (Optional) Overriding helpers like ``build_clmatrix`` when custom data structures or
   GPU kernels are required.

After these steps the subclass plugs directly into the workflow shown earlier and can be
used inside :class:`aspire.source.OrientedSource`. A simplified template is shown below:

.. code-block:: python

   from aspire.abinitio import CLMatrixOrient3D

   class CLFancySync(CLMatrixOrient3D):
       """
       Example orientation estimator built on the common-lines matrix workflow.
       """

       def estimate_rotations(self):
           # Compute and cache the clmatrix
           clmatrix = self.clmatrix

           # Custom synchronization using clmatrix statistics
           sync_mat = self._fancy_sync(clmatrix)
           est_rots = self._recover_rots(sync_mat)

	   return est_rots

       def _fancy_sync(self, clmatrix):
           # Placeholder illustrating where algorithm-specific logic would go
	   ...
           return sync_mat

       def _recover_rots(self, sync_mat):
           # Placeholder for additional algorithmic-specific logic
	   ...
	   return rots
	   
With this skeleton in place, the new class inherits masking, caching, shift estimation,
GPU dispatch hooks, and reference-free pipeline compatibility from the base classes.
