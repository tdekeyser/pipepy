"""
These pipes assume a pandas DataFrame input and produce a DataFrame output.
"""

import pandas as pd

from pipepy.core import Pipe


def category_to_numeric(vector: pd.Categorical) -> pd.Categorical:
    """
    Transform a categorical vector of data into numeric bins. The function
    returns a duplicate.

    >>> category_to_numeric(pd.Categorical(['A', 'b', 'c', 'A']))
    [0, 1, 2, 0]
    """
    uniques = [x for x in pd.unique(vector) if not pd.isna(x)]
    return pd.Categorical(vector).rename_categories(range(len(uniques)))


class CategoryToNumericPipe(Pipe):
    """
    Transform specified columns of a dataset into bins. Does not create
    a duplicate.
    """

    def __init__(self, cols):
        self.cols = cols

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            data = data.drop(col, axis='columns')
        return data
