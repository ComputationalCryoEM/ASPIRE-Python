from unittest import TestCase

from aspyre.io.star import StarFile

import os.path

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')

class NewStarFileTestcase(TestCase):
    def setUp(self):
        self.fn = os.path.join(DATA_DIR, 'rib80s_run003_model_trunc.star')

    def tearDown(self):
        pass

    def testRead(self):
        star = StarFile()
        star.read(self.fn)

        self.assertTrue('model_general' in star.sections)
        self.assertTrue('model_class_1' in star.sections)
