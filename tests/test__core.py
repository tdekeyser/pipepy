import unittest

from pipepy.core import PipeLine
from pipepy.generic_pipe import IdentityPipe, identity


def add_testpipe(data):
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
        pipeline = PipeLine([identity, add_testpipe])

        self.assertEqual(pipeline.flush(self.data), list(range(11))[1:])

    def test_flush_dataIsNone_error(self):
        pipeline = PipeLine([identity])

        with self.assertRaises(AssertionError):
            pipeline.flush(None)
