from abc import abstractmethod, ABC
from functools import reduce


class Pipe(ABC):
    """Basic interface for a pipe class. Choosing a class over a function
    allows more complex constructions such as holding state, adding
    multiple arguments, or even perform feature engineering.
    """

    @abstractmethod
    def flush(self, data):
        pass


class PipeLine(object):
    """A pipeline is defined as an ordered list of Pipes or functions that
    consecutively calls each function on data in order to transform the input.
    We can use this concept to clean or restructure a dataset after a number
    of cleaning steps.

    A pipe function is expected to have a single argument and a single output,
    i.e. the data and the transformed data.

    In the initial design, this class was not designed to be subclassed. It
    should only be used to collect pipes and call the ``flush`` function on
    the data.
    """

    def __init__(self, pipes):
        self.pipes = pipes

    def flush(self, data):
        return reduce(lambda result, pipe: self.__flush(result, pipe), self.pipes, data)

    def __flush(self, data, pipe):
        assert data is not None, "Cannot transform null"

        if isinstance(pipe, Pipe):
            return pipe.flush(data)

        elif callable(pipe):
            return pipe(data)

        raise RuntimeError('pipe needs to be callable: ' + pipe)
