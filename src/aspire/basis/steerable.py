import logging

import numpy as np

from aspire.basis import Basis
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


class SteerableBasis2D(Basis):
    """
    SteerableBasis2D is an extension of Basis that is expected to have
    `rotation` (steerable) and `calculate_bispectrum` methods.
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

        # Check shape
        if complex_coef.shape[0] != 1:
            raise ValueError(
                "Due to potentially large sizes, bispectrum is limited to a single set of coefs."
                f"  Passed shape {complex_coef.shape}"
            )

        if complex_coef.shape[1] != self.complex_count:
            raise ValueError(
                "Basis.calculate_bispectrum coefs expected"
                f" to have (complex) count {self.complex_count}, received {complex_coef.shape}."
            )

        # From here just treat complex_coef as 1d vector instead of 1 by count 2d array.
        complex_coef = complex_coef[0]

        if freq_cutoff and freq_cutoff > np.max(self.complex_angular_indices):
            logger.warning(
                f"Bispectrum frequency cutoff {freq_cutoff} outside max {np.max(self.complex_angular_indices)}"
            )

        # Notes, regarding the naming:
        # radial freq indices q in paper/slides, _indices["ks"] in code
        radial_indices = self.complex_radial_indices  # q
        # angular freq indices k in paper/slides, _indices["ells"] in code
        angular_indices = self.complex_angular_indices  # k
        # Compute the set of all unique q in the compressed basis
        #   Note that np.unique is also sorted.
        unique_radial_indices = np.unique(radial_indices)

        # When compressed, we need maps between the basis and uncompressed set of q
        #   to a reduced set of q that remain after compression.
        # One map is provided by self.complex_radial_indices
        #   which maps an index in the basis remaining after compression to a q value.
        # The following code computes a similar but inverted map,
        #   given a q value, find an index into the set of unique q after compression.
        # Also, it is known that the set of q gets sparser with increasing k
        #   but that's ignored here, instead construct a dense
        #   array and filter it later.
        #  The plan is to revisit this code after appropriate coef classes are derived.

        # Default array to fill_value, we can use a value
        # k should never achieve..
        fill_value = self.complex_count ** 2
        compressed_radial_map = (
            np.ones(np.max(unique_radial_indices) + 1, dtype=int) * fill_value
        )
        for uniq_q_index, q_value in enumerate(unique_radial_indices):
            compressed_radial_map[q_value] = uniq_q_index

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
                    # Get the specific q indices related to feasible k3 angular_indices
                    Q3_ind = radial_indices[intermodulated_coef_inds]

                    if hasattr(self, "compressed") and self.compressed:
                        # Map those Q3_ind values to indices into compressed unique_radial_indices
                        #  by using the compressed_radial_map prepared above.
                        Q3_ind = compressed_radial_map[Q3_ind]

                    Coef3 = complex_coef[intermodulated_coef_inds]

                    B[ind1, ind2, Q3_ind] = coef1 * coef2 * np.conj(Coef3)

        if filter_nonzero_freqs:
            non_zero_freqs = angular_indices != 0
            B = B[non_zero_freqs][:, non_zero_freqs]

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
            if len(radians) != len(complex_coef):
                raise RuntimeError(
                    "`rotate` call `radians` length cannot broadcast with"
                    f" `complex_coef` {len(complex_coef)} != {len(radians)}"
                )
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
