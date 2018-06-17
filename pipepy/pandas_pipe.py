"""
These pipes assume a pandas DataFrame input and produce a DataFrame output.
"""
import pandas as pd

from pipepy.core import Pipe, ResidueMixin


def category_to_numeric(vector: pd.Categorical) -> (pd.Categorical, pd.Categorical):
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

    TODO: Decide whether to use LabelEncoder(). Performance??
    """
    uniques = [x for x in pd.unique(vector) if not pd.isna(x)]
    return pd.Categorical(vector).rename_categories(range(len(uniques))), uniques


class CategoryToNumericPipe(ResidueMixin, Pipe):
    """
    Transform specified columns of a dataset into bins. Does not create
    a duplicate.

    :param cols: iterable of column names or indices
    """

    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.columns:
            data[col], residue = category_to_numeric(data[col])
            self.add_residue((col, residue))
        return data


class DropColumnPipe(Pipe):
    """
    Drop columns from the DataFrame.

    :param cols: iterable of column names or indices
    """

    def __init__(self, columns):
        self.columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.columns:
            data = data.drop(col, axis='columns')
        return data
