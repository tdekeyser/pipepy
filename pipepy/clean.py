from typing import Sequence, Callable

import pandas as pd

from pipepy.core import ResidueMixin, Pipe


class DropColumnPipe(ResidueMixin, Pipe):
    """
    Drop columns from the DataFrame.

    :param cols: iterable of column names or indices
    :param keep: If True, drops all except cols
    """

    def __init__(self, columns: Sequence, keep=False):
        super().__init__()
        self.__columns = columns
        self.__keep = keep

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.__filter_columns(data.columns):
            self.add_residue(data[col])
            data = data.drop(col, axis='columns')
        return data

    def __filter_columns(self, data_cols):
        if self.__keep:
            return list(filter(lambda col: col not in self.__columns, data_cols))
        return self.__columns


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

    `.residue` field contains the categories of the transformed columns.

    :param cols: iterable of column names or indices. If cols is None,
    all columns will be transformed.
    """

    def __init__(self, columns: Sequence = None, excludes=False):
        super().__init__()
        self.__columns = columns
        self.__excludes = excludes

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.__filter_columns(data.columns):
            data[col], residue = category_to_numeric(data[col])
            self.add_residue((col, residue))
        return data

    def __filter_columns(self, data_cols):
        if self.__excludes:
            return list(filter(lambda col: col not in self.__columns, data_cols))
        return self.__columns if self.__columns is not None else data_cols


class VariableToBinPipe(ResidueMixin, Pipe):
    """
    Transform a continuous variable into discrete intervals (bins).
    If columns is None, all columns are transformed.

    `.residue` field contains the categories of the transformed
    variables.
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
            self.add_residue((col, categories))
        return data


def is_three_std_from_mean(series):
    return (series - series.mean()).abs() > 3 * series.std()


class RemoveOutliersPipe(ResidueMixin, Pipe):
    """
    Remove rows that contain outliers, i.e. values greater than 3*std.

    `.residue` field contains indices of removed row indices.

    :param columns: iterable of column names or indices. If cols is None,
    all columns will be transformed.
    """

    def __init__(self, columns: Sequence = None, outlier_metric: Callable = is_three_std_from_mean):
        assert outlier_metric is not None, 'Outlier metric cannot be None'

        super().__init__()
        self.__columns = columns
        self.__is_outlier = outlier_metric

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = self.__columns if self.__columns is not None else data.columns
        for col in columns:
            outlier_index = data[self.__is_outlier(data[col])]
            self.add_residue(outlier_index.index)
            data = data.drop(outlier_index.index)
        return data

    def add_residue(self, residue):
        self.residue = residue.union(self.residue)
