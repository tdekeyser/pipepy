import unittest

from pipepy.core import Pipe, PipeLine


class IdentityPipe(Pipe):
    def flush(self, data):
        return data


def identity(data):
    return data


def add(data):
    return list(map(lambda x: x + 1, data))


class PipeLineTest(unittest.TestCase):

    def setUp(self):
        self.data = list(range(10))

    def test_flush_pipes(self):
        pipeline = PipeLine([IdentityPipe()])

        self.assertEqual(pipeline.flush(self.data), self.data)

    def test_flush_functions(self):
        pipeline = PipeLine([identity])

        self.assertEqual(pipeline.flush(self.data), self.data)

    def test_flush_pipes_and_functions(self):
        pipeline = PipeLine([IdentityPipe(), identity])

        self.assertEqual(pipeline.flush(self.data), self.data)

    def test_flush_initial_value(self):
        pipeline = PipeLine([identity, add])

        self.assertEqual(pipeline.flush(self.data), list(range(11))[1:])
