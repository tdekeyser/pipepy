"""
These pipes assume a pandas DataFrame input and produce a DataFrame output.
"""

from pipepy.core import Pipe


def identity(data):
    return data


class IdentityPipe(Pipe):
    def flush(self, data):
        return data


def strip(data):
    """Remove whitespace before and after.
    """
    return list(map(lambda x: x.strip(), data))


class StripPipe(Pipe):

    def __init__(self, cols=None):
        self.cols = cols

    def flush(self, data):
        return strip(data)
