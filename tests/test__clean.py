import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from pipepy.clean import category_to_numeric, CategoryToNumericPipe, DropColumnPipe, RemoveOutliersPipe


class DropColumnPipeTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]))

    def test_drop(self):
        pipe = DropColumnPipe([0, 2, 3])

        assert_array_equal(pipe.flush(self.df), np.asarray([
            [1],
            [5]
        ]))

    def test_use_colnames(self):
        self.df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]), columns=['one', 'two', 'three', 'four'])
        pipe = DropColumnPipe(['one', 'three'])

        assert_array_equal(pipe.flush(self.df), np.asarray([
            [1, 3],
            [5, 7]
        ]))

    def test_keep(self):
        pipe = DropColumnPipe([0, 2, 3], keep=True)

        assert_array_equal(pipe.flush(self.df), np.asarray([
            [0, 2, 3],
            [4, 6, 7]
        ]))


class CategoryToNumericPipeTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [0, 5, 6, 7]
        ]))

    def test_category_to_numeric(self):
        transformed_array, residue = category_to_numeric(pd.Categorical(['A', 'b', 'c', 'A']))

        assert_array_equal(transformed_array, pd.Categorical([0, 1, 2, 0]))
        self.assertEqual(residue, ['A', 'b', 'c'])

    def test_pipe(self):
        pipe = CategoryToNumericPipe([0, 2])

        assert_array_equal(pipe.flush(self.df), np.asarray([
            [0, 1, 0, 3],
            [0, 5, 1, 7]
        ]))

    def test_use_colnames(self):
        self.df = pd.DataFrame(np.asarray([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]), columns=['one', 'two', 'three', 'four'])
        pipe = CategoryToNumericPipe(['one', 'three'])

        assert_array_equal(pipe.flush(self.df), np.asarray([
            [0, 1, 0, 3],
            [1, 5, 1, 7]
        ]))

    def test_no_input_transforms_all(self):
        pipe = CategoryToNumericPipe()

        assert_array_equal(pipe.flush(self.df), np.asarray([
            [0, 0, 0, 0],
            [0, 1, 1, 1]
        ]))

    def test_excludes(self):
        pipe = CategoryToNumericPipe([2], excludes=True)

        assert_array_equal(pipe.flush(self.df), np.asarray([
            [0, 0, 2, 0],
            [0, 1, 6, 1]
        ]))

    def test_residue(self):
        pipe = CategoryToNumericPipe([0, 2])

        pipe.flush(self.df)

        self.assertEqual(pipe.residue, [(0, [0]), (2, [2, 6])])


class RemoveOutliersPipeTest(unittest.TestCase):

    def test_removes_outliers(self):
        df = pd.DataFrame(np.zeros((100, 3)))
        df.iloc[10, 0] = 10000
        pipe = RemoveOutliersPipe()

        result = pipe.flush(df)

        self.assertEqual(result.shape, (99, 3))
        assert_array_equal(result.sum(), np.asarray([0, 0, 0]))
        assert_array_equal(pipe.residue[0], np.asarray([[10000, 0, 0]]))


class VariableToBinPipeTest(unittest.TestCase):
    pass  # TODO
