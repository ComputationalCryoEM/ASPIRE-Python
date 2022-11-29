Image Formation Model in ASPIRE
===============================

Let :math:`\phi: \mathbb{R}^3 \mapsto \mathbb{R}` represent the potential of a molecule. By convention, the clean projection of the :math:`i`'th particle image is modeled as:

.. math::
   I_i (x,y) = H \ast \int_{0}^{\infty} \phi (R_i^T r) dz, \quad r = (x,y,z)^T

Where :math:`H` is the CTF and :math:`R_i \in SO(3)` the particle's orientation.
