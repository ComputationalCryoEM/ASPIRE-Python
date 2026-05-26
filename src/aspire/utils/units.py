"""
Miscellaneous utilities for common unit conversions.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def ratio_to_decibel(p):
    """
    Convert a ratio of powers to decibel (log) scale.

    Follows numpy broadcasting rules.

    :param p: Power ratio.
    :returns: Power ratio in log scale
    """

    return 10 * np.log10(p)


def voltage_to_wavelength(voltage):
    """
    Convert from electron voltage to wavelength.

    :param voltage: float, The electron voltage in kV.
    :return: float, The electron wavelength in angstroms.
    """
    # We use de Broglie's relativistic formula for wavelength given by:
    # wavelength = h / np.sqrt(2 * m * q * V * (1 + q * V / (2 * m * c**2))),
    # where
    # h = float(6.62607015e-34) is Planck's constant
    # q = float(1.602176634e-19) is elementary charge
    # m = float(9.1093837015e-31) is electron mass
    # c = float(299792458) is speed of light

    # We precalculate a = 1e10 * a / np.sqrt(2*m*q) and b = 1e6 * q / (2*m*c^2).
    # 1e10 and 1e6 are conversions from meters to angstroms and volts to kilovolts, respectively.
    a = float(12.264259661581491)
    b = float(0.9784755917869367)

    return a / math.sqrt(voltage * 1e3 + b * voltage**2)


def wavelength_to_voltage(wavelength):
    """
    Convert from electron voltage to wavelength.

    :param wavelength: float, The electron wavelength in angstroms.
    :return: float, The electron voltage in kV.
    """
    a = float(12.264259661581491)
    b = float(0.9784755917869367)

    return (-1e3 + math.sqrt(1e6 + 4 * a**2 * b / wavelength**2)) / (2 * b)
