import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from pipepy.pandas_pipe import category_to_numeric, CategoryToNumericPipe, DropColumnPipe


class CategoryToNumericPipeTest(unittest.TestCase):

    def test_category_to_numeric(self):
        assert_array_equal(
            category_to_numeric(pd.Categorical(['A', 'b', 'c', 'A'])),
            pd.Categorical([0, 1, 2, 0]))

    def test_category_to_numeric_pipe(self):
        df = pd.DataFrame(np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]))
        pipe = CategoryToNumericPipe([0, 2])

        assert_array_equal(pipe.flush(df.copy()), np.asarray([[0, 1, 0, 3], [1, 5, 1, 7]]))

    def test_category_to_numeric_pipe_colnames(self):
        df = pd.DataFrame(np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]), columns=['one', 'two', 'three', 'four'])
        pipe = CategoryToNumericPipe(['one', 'three'])

        assert_array_equal(pipe.flush(df.copy()), np.asarray([[0, 1, 0, 3], [1, 5, 1, 7]]))


class DropColumnPipeTest(unittest.TestCase):

    def test_drop_columns(self):
        df = pd.DataFrame(np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]))
        pipe = DropColumnPipe([0, 2, 3])

        assert_array_equal(pipe.flush(df.copy()), np.asarray([[1], [5]]))

    def test_drop_columns_colnames(self):
        df = pd.DataFrame(np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]), columns=['one', 'two', 'three', 'four'])
        pipe = DropColumnPipe(['one', 'three'])

        assert_array_equal(pipe.flush(df.copy()), np.asarray([[1, 3], [5, 7]]))