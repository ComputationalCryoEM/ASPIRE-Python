from unittest import TestCase

import numpy as np

from aspire.utils import voltage_to_wavelength, wavelength_to_voltage

rng = np.random.default_rng()


class ImagingTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testVoltageConversion(self):
        voltage = rng.random()
        wavelength = voltage_to_wavelength(voltage)
        self.assertAlmostEqual(voltage, wavelength_to_voltage(wavelength))

    def testWavelengthConversion(self):
        wavelength = rng.random()
        voltage = wavelength_to_voltage(wavelength)
        self.assertAlmostEqual(wavelength, voltage_to_wavelength(voltage))
