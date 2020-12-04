import os.path
from unittest import TestCase

import numpy as np

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class MySimulation(Simulation):
    # A subclassed ImageSource object that specifies a metadata alias
    metadata_aliases = {"greeting": "my_greeting"}


class SimTestCase(TestCase):
    def setUp(self):
        self.sim = MySimulation(
            n=1024,
            L=8,
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
        )

    def tearDown(self):
        pass

    def testMetadata1(self):
        # A new metadata column 'greeting' added to all images in the simulation, with the value 'hello'
        self.sim.set_metadata("greeting", "hello")
        # Get value of a metadata field for all images
        values = self.sim.get_metadata("greeting")
        # We get back 'hello' 1024 times
        self.assertTrue(np.all(np.equal(np.repeat("hello", 1024), values)))

    def testMetadata2(self):
        # Same as above, except that we set metadata twice in a row
        self.sim.set_metadata("greeting", "hello")
        self.sim.set_metadata("greeting", "goodbye")
        # Get value of a metadata field for all images
        values = self.sim.get_metadata("greeting")
        # We get back 'hello' 1024 times
        self.assertTrue(np.all(np.equal(np.repeat("goodbye", 1024), values)))

    def testMetadata3(self):
        # A new metadata column 'rand_value' added to all images in the simulation, with random values
        rand_values = np.random.rand(1024)
        self.sim.set_metadata("rand_value", rand_values)
        # Get value of a metadata field for all images
        values = self.sim.get_metadata("rand_value")
        self.assertTrue(np.allclose(rand_values, values))

    def testMetadata4(self):
        # 2 new metadata columns 'rand_value1'/'rand_value2' added, with random values
        rand_values1 = np.random.rand(1024)
        rand_values2 = np.random.rand(1024)
        new_data = np.column_stack([rand_values1, rand_values2])
        self.assertFalse(self.sim.has_metadata(["rand_value1", "rand_value2"]))
        self.sim.set_metadata(["rand_value1", "rand_value2"], new_data)
        self.assertTrue(self.sim.has_metadata(["rand_value2", "rand_value1"]))
        # Get value of metadata fields for all images
        values = self.sim.get_metadata(["rand_value1", "rand_value2"])
        self.assertTrue(np.allclose(new_data, values))

    def testMetadata5(self):
        # 2 new metadata columns 'rand_value1'/'rand_value2' added, for SPECIFIC indices
        values1 = [11, 12, 13]
        values2 = [21, 22, 23]
        new_data = np.column_stack([values1, values2])
        # Set value of metadata fields for indices 0, 1, 3
        self.sim.set_metadata(
            ["rand_value1", "rand_value2"], new_data, indices=[0, 1, 3]
        )
        # Get value of metadata fields for indices 0, 1, 2, 3
        values = self.sim.get_metadata(["rand_value1", "rand_value2"], [0, 1, 2, 3])
        self.assertTrue(
            np.allclose(
                np.column_stack([[11, 12, np.nan, 13], [21, 22, np.nan, 23]]),
                values,
                equal_nan=True,
            )
        )
