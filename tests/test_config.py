from unittest import TestCase
from aspyre.utils.config import Config, ConfigArgumentParser


class ConfigTest(TestCase):
    def setUp(self):
        # A config object is initialized by a valid ini string
        self.config = Config("""
            [math]
            greeting = Hello Math
            zero = 0
            pi = 3.1415
            names = Leibniz, Ramanujan
        """)

    def tearDown(self):
        pass

    def testString(self):
        self.assertEqual('Hello Math', self.config.math.greeting)

    def testInt(self):
        self.assertEqual(0, self.config.math.zero)

    def testFloat(self):
        self.assertAlmostEqual(3.1415, self.config.math.pi)

    def testList(self):
        self.assertEqual('Ramanujan', self.config.math.names[1])

    def testConfigArgumentParser(self):
        # A ConfigArgumentParser can be instantiated from a Config object
        # which provides an override mechanism for the Config object through a context manager

        # If the 'config' kwarg is unspecified in the constructor, the 'config' object in the aspyre package
        # is (temporarily) overridden.
        # This allows scripts to support all options that are found in the aspyre 'config' object

        # Here we test our custom 'self.config' object since we can't make any guarantees about keys present in
        # the aspyre config object
        parser = ConfigArgumentParser(config=self.config)

        # 'zero' has the expected value here
        self.assertEqual(0, self.config.math.zero)

        # A ConfigParser adds configuration parameters as 'config.<section>.<key>' options
        with parser.parse_args(['--config.math.zero', '42']):
            # value overridden - notice that the type is inferred from the original value
            self.assertTrue(int, type(self.config.math.zero))
            self.assertEqual(42, self.config.math.zero)

        # 'zero' reverts to its expected value here
        self.assertEqual(0, self.config.math.zero)
