"""
Tests that perform actual data cleaning on the datasets in the ``data/`` folder.
"""
import unittest

import pandas as pd

from pipepy.core import PipeLine

TITANIC = 'data/titanic/train.csv'


class PipeIntegrationTest(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv(TITANIC, keep_default_na=False)

    def test_case_filter_female(self):
        def filter_female(data):
            return data.pipe(lambda x: x[x['Sex'] == 'female'])

        pipeline = PipeLine([filter_female])

        print(pipeline.flush(self.data))
