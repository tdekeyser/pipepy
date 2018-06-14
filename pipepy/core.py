from abc import abstractmethod, ABC
from functools import reduce


class Pipe(ABC):

    @abstractmethod
    def flush(self, data):
        pass



class PipeLine(object):

    def __init__(self, pipes):
        self.pipes = pipes

    def flush(self, data):
        return reduce(lambda result, pipe: self.__flush(result, pipe), self.pipes, data)

    def __flush(self, data, pipe):
        if isinstance(pipe, Pipe):
            return pipe.flush(data)

        elif callable(pipe):
            return pipe(data)

        raise RuntimeError('pipe needs to be callable: ' + pipe)
