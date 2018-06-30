from typing import Callable, Sequence

import pandas as pd

from pipepy.core import Pipe


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

    def __init__(self, map_func: Callable, columns: Sequence = None):
        assert callable(map_func), 'Apply func should be callable %s' % map_func

        self.__map_func = map_func
        self.__columns = columns

    def flush(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = self.__columns if self.__columns is not None else data.columns
        for col in columns:
            data[col] = data[col].apply(self.__map_func)
        return data
