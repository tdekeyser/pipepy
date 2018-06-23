from typing import Sequence

import pandas as pd

from pipepy.core import ResidueMixin, Pipe


class DropColumnPipe(ResidueMixin, Pipe):
    """
    Drop columns from the DataFrame.

    :param cols: iterable of column names or indices
    """

    def __init__(self, columns: Sequence):
        super().__init__()
        self.__columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.__columns:
            self.add_residue(data[col])
            data = data.drop(col, axis='columns')
        return data


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

    def __init__(self, bins: int = 3, columns: Sequence = None):
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
