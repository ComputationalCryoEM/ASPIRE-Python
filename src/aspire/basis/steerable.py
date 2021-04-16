import logging

import numpy as np
from tqdm import tqdm

from aspire.basis import Basis
from aspire.utils import complex_type, real_type

logger = logging.getLogger(__name__)


class SteerableBasis(Basis):
    """
    SteerableBasis are an extension of Basis that is expected to have rotation (steerable) and bispectrum.
    """

    def calculate_bispectrum(
        self, complex_coef, flatten=False, filter_nonzero_freqs=False
    ):
        """
        Calculate bispectrum for a set of coefs in this basis.

        The Bispectum matrix is of shape:
            (count, count, unique_radial_indices)

        where count is the number of complex coefficients.

        :param coef: Coefficients representing a (single) image expanded in this basis.
        :param flatten: Optionally extract symmetric values (tril) and then flatten.
        :param filter_nonzero_freqs: Remove indices corresponding to zero frequency (defaults False).
        :return: Bispectum matrix (complex valued).
        """

        # Coefs should be 0 or 1D, remove any dim shapes equal to one, then check.
        complex_coef = np.squeeze(complex_coef)

        # Check shape
        if complex_coef.ndim != 1:
            raise ValueError(
                "Due to potentially large sizes, bispectrum is limited to a single set of coefs."
                f"  Passed shape {complex_coef.shape}"
            )

        if len(complex_coef) != self.complex_count:
            raise ValueError(
                "Basis.calculate_bispectrum coefs expected"
                f"to have (complex) count {self.complex_count}, received {len(complex_coef)}."
            )

        # self._indices["ells"]  # angular freq indices k in paper/slides
        # self._indices["ks"]    # radial freq indices q in paper/slides
        # radial_indices = self._indices["ks"][self.indices_real]  # q
        # angular_indices = self._indices["ells"][self.indices_real]  # k
        radial_indices = self.complex_radial_indices  # q
        angular_indices = self.complex_angular_indices  # k
        unique_radial_indices = np.unique(radial_indices)  # q
        unique_angular_indices = np.unique(angular_indices)  # k

        B = np.zeros(
            (self.complex_count, self.complex_count, unique_radial_indices.shape[0]),
            dtype=complex_type(self.dtype),
        )

        logger.info(f"Calculating bispectrum matrix with shape {B.shape}.")

        for ind1 in tqdm(range(self.complex_count)):

            k1 = angular_indices[ind1]
            coef1 = complex_coef[ind1]

            for ind2 in range(self.complex_count):

                k2 = angular_indices[ind2]
                coef2 = complex_coef[ind2]

                k3 = k1 + k2
                intermodulated_coef_inds = angular_indices == k3

                if np.any(intermodulated_coef_inds):
                    Q3 = radial_indices[intermodulated_coef_inds]
                    Coef3 = complex_coef[intermodulated_coef_inds]

                    B[ind1, ind2, Q3] = coef1 * coef2 * np.conj(Coef3)

        if filter_nonzero_freqs:
            non_zero_freqs = angular_indices != 0
            B = B[non_zero_freqs][:, non_zero_freqs]
        # #Normalize B ?
        # B /= np.linalg.norm(B, axis=-1)[:,:,np.newaxis]

        # dirty plot for debugging, can rm later.
        # import matplotlib.pyplot as plt
        # for q in range(B.shape[-1]):
        #     print(np.max(np.abs(B[...,q])))
        # plt.imshow(np.log(np.abs(
        #     B.reshape(B.shape[0], -1) )))
        # plt.show()

        if flatten:
            # B is sym, start by taking lower triangle.
            tril = np.tri(B.shape[0], dtype=bool)
            B = B[tril, :]
            # Then flatten
            B = B.flatten()

        return B

    def rotate(self, complex_coef, radians):
        """
        Returns complex coefs rotated by `radians`.

        :param complex_coef: Basis coefs (in complex representation).
        :param radians: Rotation in radians.
        :return: rotated (complex) coefs.
        """

        ks = self.complex_angular_indices
        rotated_coef = complex_coef * np.exp(-1j * ks * radians)

        return rotated_coef
