from unittest import TestCase

from aspire.config import Config, config_override


class ConfigTest(TestCase):
    def setUp(self):
        # A config object is initialized by a valid ini string
        self.config = Config(
            """
            [math]
            greeting = Hello Math
            zero = 0
            pi = 3.1415
            names = Leibniz, Ramanujan
        """
        )

    def tearDown(self):
        pass

    def testString(self):
        self.assertEqual("Hello Math", self.config.math.greeting)

    def testInt(self):
        self.assertEqual(0, self.config.math.zero)

    def testFloat(self):
        self.assertAlmostEqual(3.1415, self.config.math.pi)

    def testList(self):
        self.assertEqual("Ramanujan", self.config.math.names[1])

    def testConfigOverride(self):
        # We have an override mechanism for the Config object through a context manager

        # 'zero' has the expected value here
        self.assertEqual(0, self.config.math.zero)

        # If the second ('config') argument is unspecified in 'config_override', the 'config' object in the aspire
        # package is (temporarily) overridden.
        # Here we test our custom 'self.config' object since we can't make any guarantees about keys present in
        # the aspire config object.
        with config_override({"math.zero": 42}, self.config):
            # value overridden!
            self.assertEqual(42, self.config.math.zero)

        # 'zero' reverts to its expected value here
        self.assertEqual(0, self.config.math.zero)
