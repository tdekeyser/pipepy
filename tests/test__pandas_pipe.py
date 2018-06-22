import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from pipepy.pandas_pipe import category_to_numeric, CategoryToNumericPipe, DropColumnPipe, MapColumnPipe, AddColumnPipe


class CategoryToNumericPipeTest(unittest.TestCase):

    def test_category_to_numeric(self):
        transformed_array, residue = category_to_numeric(pd.Categorical(['A', 'b', 'c', 'A']))

        assert_array_equal(transformed_array, pd.Categorical([0, 1, 2, 0]))
        self.assertEqual(residue, ['A', 'b', 'c'])

    def test_category_to_numeric_pipe(self):
        df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]))
        pipe = CategoryToNumericPipe([0, 2])

        assert_array_equal(pipe.flush(df), np.asarray([
            [0, 1, 0, 3],
            [1, 5, 1, 7]
        ]))

    def test_category_to_numeric_pipe_colnames(self):
        df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]), columns=['one', 'two', 'three', 'four'])
        pipe = CategoryToNumericPipe(['one', 'three'])

        assert_array_equal(pipe.flush(df), np.asarray([
            [0, 1, 0, 3],
            [1, 5, 1, 7]
        ]))

    def test_category_to_numeric_pipe_residue(self):
        df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]))
        pipe = CategoryToNumericPipe([0, 2])

        pipe.flush(df)

        self.assertEqual(pipe.residue, [(0, [0, 4]), (2, [2, 6])])


class DropColumnPipeTest(unittest.TestCase):

    def test_drop_columns(self):
        df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]))
        pipe = DropColumnPipe([0, 2, 3])

        assert_array_equal(pipe.flush(df), np.asarray([
            [1],
            [5]
        ]))

    def test_drop_columns_colnames(self):
        df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]), columns=['one', 'two', 'three', 'four'])
        pipe = DropColumnPipe(['one', 'three'])

        assert_array_equal(pipe.flush(df), np.asarray([
            [1, 3],
            [5, 7]
        ]))


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


class ApplyOnAxisPipeTest(unittest.TestCase):

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


class VariableToBinPipeTest(unittest.TestCase):
    pass # TODO
