import unittest

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
        pipe = MapColumnPipe(lambda col: col + 1, columns=['one', 'four'])

        assert_array_equal(pipe.flush(df), np.asarray([
            [1, 1, 2, 4],
            [5, 5, 6, 8]
        ]))
