"""
Tests that perform actual data cleaning on the datasets in the ``data/`` folder.
"""
import pandas as pd
import unittest

from pipepy.core import PipeLine
from pipepy.pipe import strip

TITANIC = 'data/titanic_train.csv'


class PipeIntegrationTest(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv(TITANIC, keep_default_na=False)

    def test_clean(self):
        pipeline = PipeLine([strip])

        pipeline.flush(self.data)

        print(self.data)
        ## TODO
