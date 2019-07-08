from random import random
from unittest import TestCase

from aspyre.utils.em import voltage_to_wavelength, wavelength_to_voltage


class ImagingTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testVoltageConversion(self):
        voltage = random()
        wavelength = voltage_to_wavelength(voltage)
        self.assertAlmostEqual(voltage, wavelength_to_voltage(wavelength))

    def testWavelengthConversion(self):
        wavelength = random()
        voltage = wavelength_to_voltage(wavelength)
        self.assertAlmostEqual(wavelength, voltage_to_wavelength(voltage))
