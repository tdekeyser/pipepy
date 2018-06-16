from pipepy.core import Pipe


def peek(data):
    print(data)
    return data


def identity(data):
    return data


class IdentityPipe(Pipe):
    def flush(self, data):
        return data
