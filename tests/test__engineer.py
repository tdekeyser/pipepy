import unittest

import time
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from pipepy.engineer import AddColumnPipe, MapColumnPipe


class AddColumnPipeTest(unittest.TestCase):

    def test_add_column(self):
        df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]))
        pipe = AddColumnPipe(lambda data: data.iloc[:, 0] + 10, 'one')

        result = pipe.flush(df)

        self.assertIn('one', result.columns)
        assert_array_equal(result, np.asarray([
            [0, 1, 2, 3, 10],
            [4, 5, 6, 7, 14]
        ]))


class MapColumnPipeTest(unittest.TestCase):

    def test_apply_on_axis(self):
        df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]), columns=['one', 'two', 'three', 'four'])
        pipe = MapColumnPipe(lambda col: col + col.mean(), columns=['one', 'four'])

        assert_array_equal(pipe.flush(df), np.asarray([
            [2, 1, 2, 8],
            [6, 5, 6, 12]
        ]))

    def test_performance(self):
        df = pd.DataFrame(np.random.randn(5000, 5000))
        pipe = MapColumnPipe(lambda col: 0 + col.mean())

        start = time.time()
        df = pipe.flush(df)
        end = time.time()

        self.assertLess(end - start, 2)
        self.assertEqual(len(pd.unique(df.values.ravel())), 5000)
