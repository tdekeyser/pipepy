from typing import Callable

import pandas as pd

from pipepy.core import Pipe


def identity(data):
    return data


class IdentityPipe(Pipe):
    def flush(self, data):
        return data


def peek(data):
    print(data)
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
