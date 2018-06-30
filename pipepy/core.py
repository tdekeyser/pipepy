from abc import abstractmethod, ABC
from functools import reduce
from typing import Callable, TypeVar, Iterable


class Pipe(ABC):
    """
    Basic interface for a pipe class. Choosing a class over a function
    allows more complex constructions such as holding state, adding
    multiple arguments, or even perform feature engineering.
    """

    @abstractmethod
    def flush(self, data):
        pass


class ResidueMixin(ABC):
    """
    This ResidueMixin allows state to be captured and aggregated. The Mixin
    can be used in conjunction with Pipe to keep track of lost data during a
    flush.

    For example usage, see `<pipepy.pandas_pipe.CategoryToNumericPipe>`, in which
    ResidueMixin is used to hold the original categories of the transformed columns.
    """

    def __init__(self):
        self.residue = []

    def add_residue(self, residue):
        self.residue.append(residue)


class Pipeline(object):
    """
    A pipeline is defined as an ordered list of Pipes or functions that
    consecutively calls each function on data in order to transform the input.
    We can use this concept to clean or restructure a dataset after a number
    of cleaning steps.

    A pipe function is expected to have a single argument and a single output,
    i.e. the data and the transformed data.

    This class is not intended to be subclassed. It should only be used to
    collect pipes and call the ``flush`` function on the data.
    """

    Pipe_t = TypeVar('Pipe_t', Pipe, Callable)

    def __init__(self, pipes: Iterable[Pipe_t], verbose=False):
        self.pipes = pipes
        self.__verbose = verbose

    def flush(self, data):
        return reduce(lambda result, pipe: self.__flush(result, pipe), self.pipes, data)

    def __flush(self, data, pipe: Pipe_t):
        assert data is not None, 'Cannot transform null'
        self.__print_stage(pipe)

        if isinstance(pipe, Pipe):
            return pipe.flush(data)

        elif callable(pipe):
            return pipe(data)

        raise RuntimeError('pipe needs to be callable: ' + pipe)

    def __print_stage(self, stage):
        if self.__verbose:
            print('Pipe - ' + str(stage) + '\n')
