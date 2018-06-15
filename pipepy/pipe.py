"""
These pipes assume a pandas DataFrame input and produce a DataFrame output.
"""
import numpy as np
import pandas as pd

from pipepy.core import Pipe


def peek(data):
    print(data)
    return data


def identity(data):
    return data


class IdentityPipe(Pipe):
    def flush(self, data):
        return data


def category_to_numeric(vector):
    """
    Tranform a categorical vector of data into numeric bins.

    >>> category_to_numeric(pd.Categorical(['A', 'b', 'c', 'A']))
    [0, 1, 2, 0]
    """
    assert vector.dtype == 'category', "Vector should be categorical"

    uniques = [x for x in pd.unique(vector) if x is not np.nan]
    return pd.Categorical(vector).rename_categories(range(len(uniques)))


class CategoryToNumericPipe(Pipe):

    def __init__(self, cols):
        self.cols = cols

    def flush(self, data):
        for col in self.cols:
            data[col] = category_to_numeric(data[col])
        return data


class DropColumnPipe(Pipe):
    """
    Drop columns from the DataFrame.

    :param cols: iterable of column names
    """

    def __init__(self, cols):
        self.cols = cols

    def flush(self, data):
        for col in self.cols:
            data = data.drop(col, axis='columns')
        return data