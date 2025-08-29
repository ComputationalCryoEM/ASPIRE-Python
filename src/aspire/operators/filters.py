import inspect
import logging
from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from aspire import config
from aspire.utils import cart2pol, grid_2d, voltage_to_wavelength

logger = logging.getLogger(__name__)


def evaluate_src_filters_on_grid(src, indices=None):
    """
    Given an ImageSource object, compute the source's unique filters
    at the filter_indices specified in its metadata.

    :param src: Source instance
    :param indices: Optional, subset of src indices to compute.
        Defaults to the entire src.

    :return: an `src.L x src.L x len(src.filter_indices)`
        array containing the evaluated filters at each gridpoint
    """

    if indices is None:
        indices = np.arange(src.n, dtype=int)

    grid2d = grid_2d(src.L, indexing="yx", dtype=src.dtype)
    omega = np.pi * np.vstack((grid2d["x"].flatten(), grid2d["y"].flatten()))

    # Initialize h as ones to mimic an IdentityFilter when src.unique_filters is None.
    h = np.ones((omega.shape[-1], len(indices)), dtype=src.dtype)
    for i, filt in enumerate(src.unique_filters):
        idx_k = np.where(src.filter_indices[indices] == i)[0]
        if len(idx_k) > 0:
            filter_values = filt.evaluate(omega, pixel_size=src.pixel_size)
            h[:, idx_k] = np.column_stack((filter_values,) * len(idx_k))

    h = np.reshape(h, grid2d["x"].shape + (len(indices),))

    return h


# TODO: filters should probably be dtyped...
class Filter:
    def __init__(self, dim=None, radial=False):
        self.dim = dim
        self.radial = radial

    def __mul__(self, other):
        return MultiplicativeFilter(self, other)

    def __str__(self):
        """
        Show class name of Filter

        :return: A string of class name
        """
        return self.__class__.__name__

    def evaluate(self, omega, **kwargs):
        """
        Evaluate the filter at specified frequencies.

        :param omega: A vector of size n (for 1d filters), or an array of size 2-by-n, representing the spatial
            frequencies at which the filter is to be evaluated. These are normalized so that pi is equal to the Nyquist
            frequency.
        :return: The value of the filter at the specified frequencies.
        """
        if omega.ndim == 1:
            assert self.radial, "Cannot evaluate a non-radial filter on 1D input array."
        elif omega.ndim == 2 and self.dim:
            assert (
                omega.shape[0] == self.dim
            ), f"Omega must be of size {self.dim} x n; Passed omega.shape {omega.shape}"

        if self.radial:
            if omega.ndim > 1:
                omega = np.sqrt(np.sum(omega**2, axis=0))
            omega, idx = np.unique(omega, return_inverse=True)
            omega = np.vstack((omega, np.zeros_like(omega)))

        h = self._evaluate(omega, **kwargs)

        if self.radial:
            h = np.take(h, idx)

        return h

    def _evaluate(self, omega, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")

    def basis_mat(self, basis, **kwargs):
        """
        Represent the filter in `basis`.

        :param basis: 2D Basis.
        :return: `basis` representation of this filter.
            Return type will depend on `basis`.
        """
        return basis.filter_to_basis_mat(self, **kwargs)

    def scale(self, c=1):
        """
        Scale filter by a constant factor

        :param c: The scaling factor. For c < 1, it dilates the filter(s) in frequency, while for c > 1,
            it compresses (default 1).
        :return: A ScaledFilter object
        """
        return ScaledFilter(self, c)

    @lru_cache(maxsize=config["cache"]["filter_cache_size"].get())  # noqa: B019
    def evaluate_grid(self, L, *args, dtype=np.float32, **kwargs):
        """
        Generates a two dimensional grid with prescribed dtype,
        yielding the values (omega) which are then evaluated by
        the filter's evaluate method.

        Passes arbritrary args and kwargs down to self.evaluate method.

        :param L: Number of grid points (L by L).
        :param dtype: dtype of grid, defaults np.float32.
        :return: Filter values at omega's points.
        """

        grid2d = grid_2d(L, indexing="yx", dtype=dtype)
        omega = np.pi * np.vstack((grid2d["x"].flatten(), grid2d["y"].flatten()))
        h = self.evaluate(omega, *args, **kwargs)

        h = h.reshape(grid2d["x"].shape)

        return h

    def dual(self):
        return DualFilter(self)

    @property
    def sign(self):
        """
        A Filter object to evaluate the signs of the underlying filter.
        """
        return LambdaFilter(self, np.sign)


class DualFilter(Filter):
    """
    A Filter object that is dual to origin one, namely g(w)=f(-w)
    """

    def __init__(self, filter_in):
        self._filter = filter_in
        super().__init__()

    def evaluate(self, omega, **kwargs):
        return self._filter.evaluate(-omega, **kwargs)


class FunctionFilter(Filter):
    """
    A Filter object that is instantiated directly using a 1D or 2D function, which is then directly used for evaluating
    the filter.
    """

    def __init__(self, f, dim=None):
        n_args = len(inspect.signature(f).parameters)
        assert n_args in (1, 2), "Only 1D or 2D functions are supported"

        assert dim in (None, 1, 2), "Only 1D or 2D dimensions are supported"
        dim = dim or n_args

        self.f = f  # will be used directly in this Filter's evaluate method
        # Note: The function may well be radial from the caller's perspective, but we won't be applying it in a radial
        # manner if the function we were initialized from expected 2 arguments
        # (i.e. at runtime, we will still expect the incoming omega values to have x and y components).
        super().__init__(dim=dim, radial=dim > n_args)

    def _evaluate(self, omega, **kwargs):
        # Note kwargs are not used here, this might be trouble
        return self.f(*omega)


class PowerFilter(Filter):
    """
    A Filter object that is composed of a regular `Filter` object, but evaluates it to a specified power.
    """

    def __init__(self, filter, power=1, epsilon=None):
        """
        Initialize PowerFilter instance.

        :param filter: A Filter instance.
        :param power: Exponent to raise filter values.
        :param epsilon: Threshold on filter values that get raised to a negative power.
            `filter` values below this threshold will be set to zero during evaluation.
            Default uses machine epsilon for filter.dtype.
        """
        self._filter = filter
        self._power = power
        self._epsilon = epsilon
        super().__init__(dim=filter.dim, radial=filter.radial)

    def _evaluate(self, omega, **kwargs):
        return self._filter.evaluate(omega, **kwargs) ** self._power

    @lru_cache(maxsize=config["cache"]["filter_cache_size"].get())  # noqa: B019
    def evaluate_grid(self, L, *args, dtype=np.float32, **kwargs):
        """
        Calls the provided filter's evaluate_grid method in case there is an optimization.

        If no optimized method is provided, falls back to base `evaluate_grid`.

        See `Filter.evaluate_grid` for usage.
        """
        filter_vals = self._filter.evaluate_grid(L, *args, dtype=dtype, **kwargs)

        # Place safeguard on values below machine epsilon for negative powers.
        if self._power < 0:
            eps = self._epsilon
            if eps is None:
                eps = np.finfo(filter_vals.dtype).eps
                eps = (100 * eps) ** (-1 / self._power)
            condition = abs(filter_vals) < eps
            num_less_eps = np.count_nonzero(condition)
            if num_less_eps > 0:
                logger.warning(
                    f"{self} setting {num_less_eps} extremal filter value(s) to zero."
                )

            filter_vals = np.where(condition, 0, filter_vals**self._power)

            return filter_vals

        return filter_vals**self._power


class LambdaFilter(Filter):
    """
    A Filter object to evaluate lambda function of a regular `Filter`.
    """

    def __init__(self, filter, f):
        self._filter = filter
        self._f = f
        super().__init__(dim=filter.dim, radial=filter.radial)

    def _evaluate(self, omega, **kwargs):
        return self._f(self._filter.evaluate(omega, **kwargs))


class MultiplicativeFilter(Filter):
    """
    A Filter object that returns the product of the evaluation of its individual filters
    """

    def __init__(self, *args):
        super().__init__(dim=args[0].dim, radial=all(c.radial for c in args))
        self._components = args

    def _evaluate(self, omega, **kwargs):
        res = 1
        for c in self._components:
            res *= c.evaluate(omega, **kwargs)
        return res


class ScaledFilter(Filter):
    """
    A Filter object that is composed of a regular `Filter` object, but evaluates it on a scaled omega.
    """

    def __init__(self, filt, scale):
        self._filter = filt
        self._scale = scale
        super().__init__(dim=filt.dim, radial=filt.radial)

    def _evaluate(self, omega, **kwargs):
        return self._filter.evaluate(omega / self._scale, **kwargs)

    def __str__(self):
        """
        Show class name of ScaledFilter and related information

        :return: A string of class name and related information
        """
        return f"ScaledFilter (scales {self._filter} by {self._scale})"


class ArrayFilter(Filter):
    def __init__(self, xfer_fn_array):
        """
        A Filter corresponding to the filter with the specified transfer function.

        :param xfer_fn_array: The transfer function of the filter in the form of an array of one or two dimensions.
        """
        dim = xfer_fn_array.ndim
        assert dim in (1, 2), "Only dimensions 1 and 2 supported."

        super().__init__(dim=dim, radial=False)

        # sz is assigned before we do anything with xfer_fn_array
        self.sz = xfer_fn_array.shape

        # The following code, though superficially different from the MATLAB code its copied from,
        # results in the same behavior.
        # TODO: This could use documentation - very unintuitive!
        if dim == 1:
            # If we have a vector of even length, then append the first element to the last
            if xfer_fn_array.shape[0] % 2 == 0:
                xfer_fn_array = np.concatenate(
                    (xfer_fn_array, np.array([xfer_fn_array[0]]))
                )
        elif dim == 2:
            # If we have a 2d array with an even number of rows, append the first row reversed at the bottom
            if xfer_fn_array.shape[0] % 2 == 0:
                xfer_fn_array = np.vstack((xfer_fn_array, xfer_fn_array[0, ::-1]))
            # If we have a 2d array with an even number of columns, append the first column reversed at the right
            if xfer_fn_array.shape[1] % 2 == 0:
                xfer_fn_array = np.hstack(
                    (xfer_fn_array, xfer_fn_array[::-1, 0][:, np.newaxis])
                )

        self.xfer_fn_array = xfer_fn_array

    def _evaluate(self, omega, **kwargs):
        _input_pts = tuple(np.linspace(1, x, x) for x in self.xfer_fn_array.shape)

        # TODO: This part could do with some documentation - not intuitive!
        temp = np.array(self.sz)[:, np.newaxis]
        omega = (omega / (2 * np.pi)) * temp
        omega += np.floor(temp / 2) + 1

        # Emulating the behavior of interpn(V,X1q,X2q,X3q,...) in MATLAB
        # The original MATLAB was using 'linear' and zero fill.
        # We will use 'linear' but fill_value=None which will extrapolate
        #  for values slightly outside the interpolation grid bounds.
        interpolator = RegularGridInterpolator(
            _input_pts,
            # https://github.com/scipy/scipy/issues/17718
            self.xfer_fn_array.astype(np.float64),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        result = interpolator(
            # Split omega into input arrays and stack depth-wise because that's how
            # the interpolator wants it
            np.dstack(np.split(omega, len(self.sz)))
        )

        # Result is 1 x np.prod(self.sz) in shape; convert to a 1-d vector
        result = np.squeeze(result, 0)

        return result

    # No need to cache ArrayFilter
    def evaluate_grid(self, L, *args, dtype=np.float32, **kwargs):
        """
        Optimized evaluate_grid method for ArrayFilter.

        If evaluate_grid is called with a resolution L that matches
        the transfer function `xfer_fn_array` resolution,
        we do not need to generate a grid, setup interpolation, and
        evaluate by interpolation. We can instead use the transfer
        function directly.

        In the case the grid is not a match, we fall back to the
        base `evaluate_grid` implementation.

        See Filter.evaluate_grid for usage.
        """
        if all(dim == L for dim in self.xfer_fn_array.shape):
            logger.debug(
                "Size of transfer function matches evaluate_grid size L exactly,"
                " skipping grid generation and interpolation."
            )
            res = self.xfer_fn_array
        else:
            # Otherwise call parent code to generate a grid then evaluate.
            res = super().evaluate_grid(L, *args, dtype=dtype, **kwargs)
        return res


class ScalarFilter(Filter):
    def __init__(self, dim=None, value=1):
        super().__init__(dim=dim, radial=True)
        self.value = value

    def __repr__(self):
        return f"Scalar Filter (dim={self.dim}, value={self.value})"

    def _evaluate(self, omega, **kwargs):
        return self.value * np.ones_like(omega)


class ZeroFilter(ScalarFilter):
    def __init__(self, dim=None):
        super().__init__(dim=dim, value=0)


class IdentityFilter(ScalarFilter):
    def __init__(self, dim=None):
        super().__init__(dim=dim, value=1)


class CTFFilter(Filter):
    """
    Reproduce MATLAB's cryo_CTF_relion CTF (Contrast Transfer Function) Filter

    Note if comparing to legacy MATLAB cryo_CTF_Relion,
    take care regarding defocus unit conversion to/from nm.
    """

    def __init__(
        self,
        voltage=200,
        defocus_u=15000,
        defocus_v=15000,
        defocus_ang=0,
        Cs=2.26,
        alpha=0.07,
        B=0,
    ):
        """
        A CTF (Contrast Transfer Function) Filter

        Note if comparing to legacy MATLAB cryo_CTF_Relion,
        take care regarding defocus unit conversion to nm.

        :param voltage:     Electron voltage in kV
        :param defocus_u:   Defocus depth along the u-axis in angstrom
        :param defocus_v:   Defocus depth along the v-axis in angstrom
        :param defocus_ang: Angle between the x-axis and the u-axis in radians
        :param Cs:          Spherical aberration constant in mm
        :param alpha:       Amplitude contrast phase in radians
        :param B:           Envelope decay in inverse square angstrom (default 0)
        """
        super().__init__(dim=2, radial=defocus_u == defocus_v)
        self.voltage = voltage
        self.wavelength = voltage_to_wavelength(self.voltage)
        self.defocus_u = defocus_u
        self.defocus_v = defocus_v
        self.defocus_ang = defocus_ang
        self.Cs = Cs
        self.alpha = alpha
        self.B = B

        # Convert angstrom to nm and divide by 2
        self._defocus_mean_nm = 0.05 * (self.defocus_u + self.defocus_v)
        self._defocus_diff_nm = 0.05 * (self.defocus_u - self.defocus_v)

    def _evaluate(self, omega, **kwargs):
        # Ensure we have a pixel size,
        pixel_size = kwargs.get("pixel_size", None)
        if pixel_size is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.evaluate must be passed kwarg `pixel_size`."
            )
        # and that it is a floating point value.
        pixel_size = float(pixel_size)

        # Reference MATLAB code, includes reference to paper
        #    Mindell, J. A.; Grigorieff, N. (2003).
        # https://github.com/PrincetonUniversity/aspire/blob/760a43b35453e55ff2d9354339e9ffa109a25371/projections/cryo_CTF_Relion.m#L34
        #
        # s, theta should match MATLAB's RadiusNorm up to a transpose
        # To accomplish this given ASPIRE-Python's default `omega` grid,
        # we unpack and remove the pi scaling,
        # and further rescale the radii `s` by half below.
        #
        # Additionally we upcast so downstream computations remain in doubles.
        x, y = omega.astype(np.float64, copy=False) / np.pi

        # Returns radii such that when multiplied by the
        # bandwidth of the signal, we get the correct radial frequencies
        # corresponding to each pixel in our nxn grid.
        theta, s = cart2pol(x, y)
        s = s / 2

        # Wavelength in nm.
        lamb = 1.22639 / np.sqrt(self.voltage * 1000 + 0.97845 * self.voltage**2)

        # Divide by 10 to make pixel size in nm. BW is the
        # bandwidth of the signal corresponding to the given pixel size.
        BW = 1 / (pixel_size / 10)

        s = s * BW
        DFavg = self._defocus_mean_nm  # (DefocusU+DefocusV)/2
        DFdiff = self._defocus_diff_nm  # (DefocusU-DefocusV)
        # Note division by 2 is pre-computed in _defocus_diff_nm
        df = DFavg + DFdiff * np.cos(2 * (theta - self.defocus_ang))

        k2 = np.pi * lamb * df
        # 10*6 converts Cs from mm to nm.
        k4 = np.pi / 2 * 10**6 * self.Cs * lamb**3
        chi = k4 * s**4 - k2 * s**2

        h = np.sqrt(1 - self.alpha**2) * np.sin(chi) - self.alpha * np.cos(chi)

        if self.B:
            h *= np.exp(-self.B * s**2)

        return h


class RadialCTFFilter(CTFFilter):
    def __init__(self, voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0):
        super().__init__(
            voltage=voltage,
            defocus_u=defocus,
            defocus_v=defocus,
            defocus_ang=0,
            Cs=Cs,
            alpha=alpha,
            B=B,
        )


class BlueFilter(Filter):
    """
    Filter where power increases with frequency.
    """

    def __init__(self, dim=None, var=1):
        super().__init__(dim=dim, radial=True)
        self.var = var

    def __repr__(self):
        return f"BlueFilter(dim={self.dim}, var={self.var})"

    def _evaluate(self, omega, **kwargs):
        f = np.sqrt(omega[0])
        m = np.mean(f)
        f = f / m

        return self.var * f


class PinkFilter(Filter):
    """
    Filter where power decreases with frequency.
    """

    def __init__(self, dim=None, var=1):
        super().__init__(dim=dim, radial=True)
        self.var = var

    def __repr__(self):
        return f"PinkFilter(dim={self.dim}, var={self.var})"

    def _evaluate(self, omega, **kwargs):
        step = np.abs(np.subtract(*omega[0][:2]))
        # Avoid zero division
        f = np.sqrt(2 * step / (omega[0] + step))
        m = np.mean(f)
        f = f / m

        return self.var * f
