import logging

import numpy as np

from aspire.basis import Basis
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


class SteerableBasis(Basis):
    """
    SteerableBasis are an extension of Basis that is expected to have rotation (steerable) and bispectrum.
    """

    def calculate_bispectrum(
        self, complex_coef, flatten=False, filter_nonzero_freqs=False, freq_cutoff=None
    ):
        """
        Calculate bispectrum for a set of coefs in this basis.

        The Bispectum matrix is of shape:
            (count, count, unique_radial_indices)

        where count is the number of complex coefficients.

        :param coef: Coefficients representing a (single) image expanded in this basis.
        :param flatten: Optionally extract symmetric values (tril) and then flatten.
        :param filter_nonzero_freqs: Remove indices corresponding to zero frequency (defaults False).
        :param freq_cutoff: Truncate (zero) high k frequecies above (int) value, defaults off (None).
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
                f" to have (complex) count {self.complex_count}, received {len(complex_coef)}."
            )

        if freq_cutoff and freq_cutoff > np.max(self.complex_angular_indices):
            logger.warning(
                f"Bispectrum frequency cutoff {freq_cutoff} outside max {np.max(self.complex_angular_indices)}"
            )

        # self._indices["ells"]  # angular freq indices k in paper/slides
        # self._indices["ks"]    # radial freq indices q in paper/slides
        # radial_indices = self._indices["ks"][self.indices_real]  # q
        # angular_indices = self._indices["ells"][self.indices_real]  # k
        radial_indices = self.complex_radial_indices  # q
        angular_indices = self.complex_angular_indices  # k
        unique_radial_indices = np.unique(radial_indices)  # q
        # unique_angular_indices = np.unique(angular_indices)  # k
        # compressed_radial_map = {q: i for i, q in enumerate(unique_radial_indices)}
        # maybe rm this if we go with important sampling...
        if hasattr(self, "compressed") and self.compressed:
            # k should never be this high..
            fill_value = self.complex_count + 1
            compressed_radial_map = (
                np.ones(np.max(unique_radial_indices) + 1, dtype=int) * fill_value
            )
            for i, q in enumerate(unique_radial_indices):
                compressed_radial_map[q] = i

        B = np.zeros(
            (self.complex_count, self.complex_count, unique_radial_indices.shape[0]),
            dtype=complex_type(self.dtype),
        )

        logger.info(f"Calculating bispectrum matrix with shape {B.shape}.")

        for ind1 in range(self.complex_count):

            k1 = angular_indices[ind1]
            if freq_cutoff and k1 > freq_cutoff:
                continue
            coef1 = complex_coef[ind1]

            for ind2 in range(self.complex_count):

                k2 = angular_indices[ind2]
                if freq_cutoff and k2 > freq_cutoff:
                    continue
                coef2 = complex_coef[ind2]

                k3 = k1 + k2
                intermodulated_coef_inds = angular_indices == k3

                if np.any(intermodulated_coef_inds):
                    Q3 = radial_indices[intermodulated_coef_inds]

                    if hasattr(self, "compressed") and self.compressed:
                        # The compressed mapping is sparse in q
                        # Q3 = [compressed_radial_map[q] for q in Q3]
                        Q3 = compressed_radial_map[Q3]

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

    def rotate(self, complex_coef, radians, refl=None):
        """
        Returns complex coefs rotated by `radians`.

        :param complex_coef: Basis coefs (in complex representation).
        :param radians: Rotation in radians.
        :param refl: Optional reflect image (about y=x) (bool)
        :return: rotated (complex) coefs.
        """

        # Covert radians to a broadcastable shape
        if isinstance(radians, np.ndarray):
            len(radians) == len(complex_coef), f"{len(radians)} != {len(complex_coef)}"
            radians = radians.reshape(-1, 1)
        # else: radians can be a constant

        ks = self.complex_angular_indices
        assert len(ks) == complex_coef.shape[-1]

        # refl
        if refl is not None:
            if isinstance(refl, np.ndarray):
                assert len(refl) == len(complex_coef)
            # else: refl can be a constant
            # get the coefs corresponding to -ks , aka "ells"
            complex_coef[refl] = np.conj(complex_coef[refl])

        complex_coef[:] = complex_coef * np.exp(-1j * ks * radians)

        return complex_coef
