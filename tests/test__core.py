import unittest

from pipepy.core import Pipeline
from pipepy.generic_pipe import IdentityPipe, identity


def add_testpipe(data):
    return list(map(lambda x: x + 1, data))


class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.data = list(range(10))

    def test_flush_pipes(self):
        pipeline = Pipeline([IdentityPipe()])

        self.assertEqual(pipeline.flush(self.data), self.data)

    def test_flush_functions(self):
        pipeline = Pipeline([identity])

        self.assertEqual(pipeline.flush(self.data), self.data)

    def test_flush_pipes_and_functions(self):
        pipeline = Pipeline([IdentityPipe(), identity])

        self.assertEqual(pipeline.flush(self.data), self.data)

    def test_flush_initial_value(self):
        pipeline = Pipeline([identity, add_testpipe])

        self.assertEqual(pipeline.flush(self.data), list(range(11))[1:])

    def test_flush_dataIsNone_error(self):
        pipeline = Pipeline([identity])

        with self.assertRaises(AssertionError):
            pipeline.flush(None)
