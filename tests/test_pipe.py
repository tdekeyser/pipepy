import unittest

from pipepy.pipe import strip


class StripPipeTest(unittest.TestCase):

    def test_strip_function(self):
        self.assertEqual(strip(['naj  ', '  bnaj', '2']), ['naj', 'bnaj', '2'])
