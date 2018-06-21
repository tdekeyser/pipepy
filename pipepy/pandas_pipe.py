"""
These pipes assume a pandas DataFrame input and produce a DataFrame output.
"""
from typing import Callable, List

import pandas as pd

from pipepy.core import Pipe, ResidueMixin


def category_to_numeric(vector: pd.Categorical):
    """
    Transform a categorical vector of data into numeric bins. The function
    returns a duplicate and the original categories.

    >>> data = pd.Categorical(['A', 'b', 'c', 'A'])
    >>> category_to_numeric(data)
    ([0, 1, 2, 0]
    Categories (3, int64): [0, 1, 2], ['A', 'b', 'c'])

    Another option is to use a LabelEncoder from sklearn.preprocessing,
    but the current implementation avoids this dependency (although you may want
    to have sklearn for modeling nevertheless).

    >>> from sklearn.preprocessing import LabelEncoder
    >>> LabelEncoder().fit_transform(data)
    array([0, 1, 2, 0])
    """
    uniques = [x for x in pd.unique(vector) if not pd.isna(x)]
    return pd.to_numeric(pd.Categorical(vector).rename_categories(range(len(uniques)))), uniques


class CategoryToNumericPipe(ResidueMixin, Pipe):
    """
    Transform specified columns of a dataset into bins. Does not create
    a duplicate.

    :param cols: iterable of column names or indices
    """

    def __init__(self, columns):
        super().__init__()
        self.__columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.__columns:
            data[col], residue = category_to_numeric(data[col])
            self.add_residue((col, residue))
        return data


class DropColumnPipe(Pipe):
    """
    Drop columns from the DataFrame.

    :param cols: iterable of column names or indices
    """

    def __init__(self, columns):
        self.__columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.__columns:
            data = data.drop(col, axis='columns')
        return data


class AddColumnPipe(Pipe):
    """
    Add columns to data.

    :param cols: Columns to add
    :param names: Respective names for the new columns
    """

    def __init__(self, columns, names):
        assert len(columns) == len(names), \
            'Length of columns must be the same as the column names: %i - %i' % (len(columns), len(names))

        self.__columns = columns
        self.__names = names

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        num_rows = data.shape[0]

        for i, col in enumerate(self.__columns):
            assert num_rows == col.shape[0]
            data[self.__names[i]] = col

        return data


class MapColumnPipe(Pipe):
    """
    Apply function func on requested columns. If requested columns is None,
    the function will be applied on all columns.

    :param func: Function to be applied
    :param columns: Column names or indices to apply func; None means 'all columns'
    """

    def __init__(self, map_func: Callable, columns: List = None):
        assert callable(map_func), 'Apply func should be callable %s' % map_func

        self.__map_func = map_func
        self.__columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = self.__columns if self.__columns is not None else data.columns
        for col in columns:
            data[col] = self.__map_func(data[col])
        return data


class PeekPipe(Pipe):
    """
    Allows to perform actions on the data that does not alter
    it in any way. Suitable for data printing or inspection.

    This pipe always performs :peek_func: on a copy of the
    data. For very large datasets, this pipe is not recommended!

    :param peek_func: Function to apply on data
    """

    def __init__(self, peek_func: Callable):
        assert callable(peek_func), 'Peek function should be callable.'

        self.__peek_func = peek_func

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        self.__peek_func(data.copy())
        return data
