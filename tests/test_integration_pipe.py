"""
Tests that perform actual data cleaning on the datasets in the ``data/`` folder.
"""
import unittest

import pandas as pd

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

    def test_case_filter_female(self):
        def filter_female(data):
            return data.pipe(lambda x: x[x['Sex'] == 'female'])

        pipeline = PipeLine([filter_female])

        print(pipeline.flush(self.data))
