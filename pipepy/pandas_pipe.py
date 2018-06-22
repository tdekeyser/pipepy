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


class VariableToBinPipe(ResidueMixin, Pipe):
    """
    Transform a continuous variable into discrete intervals (bins).
    If columns is None, all columns are transformed.
    """

    def __init__(self, bins=3, columns=None):
        assert bins > 0, 'Cannot transform nr of bins: %i' % bins

        super().__init__()
        self.__bins = bins
        self.__columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = self.__columns if self.__columns is not None else data.columns
        for col in columns:
            data[col], categories = pd.cut(data[col], bins=self.__bins, labels=False, retbins=True)
            self.add_residue(categories)
        return data


class DropColumnPipe(ResidueMixin, Pipe):
    """
    Drop columns from the DataFrame.

    :param cols: iterable of column names or indices
    """

    def __init__(self, columns):
        super().__init__()
        self.__columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.__columns:
            self.add_residue(data[col])
            data = data.drop(col, axis='columns')
        return data


class AddColumnPipe(Pipe):
    """
    Perform feature engineering to add a column to data.

    :param engineer_funcs: Callable to engineer a new column
    :param names: Name for the new column
    """

    def __init__(self, engineer_func: Callable[[pd.DataFrame], pd.Series], name: str):
        self.__engineer_func = engineer_func
        self.__name = name

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        column = self.__engineer_func(data)

        assert data.shape[0] == column.shape[0], \
            'Engineered column should have same number of rows as input data %i - %i' % (column.shape[0], data.shape[0])

        data[self.__name] = column
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
