"""
Utility functions for Electron-Microscopy
"""

import math


def voltage_to_wavelength(voltage):
    """
    Convert from electron voltage to wavelength.
    :param voltage: float, The electron voltage in kV.
    :return: float, The electron wavelength in nm.
    """
    return 12.2643247 / math.sqrt(voltage*1e3 + 0.978466*voltage**2)


def wavelength_to_voltage(wavelength):
    """
    Convert from electron voltage to wavelength.
    :param wavelength: float, The electron wavelength in nm.
    :return: float, The electron voltage in kV.
    """
    return (-1e3 + math.sqrt(1e6 + 4 * 12.2643247**2 * 0.978466 / wavelength**2)) / (2 * 0.978466)
