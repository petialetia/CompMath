import unittest
import numpy

from NewtonInterpolation import *

class NewtonInterpolationTestCase(unittest.TestCase):
    def testDividedDifferences(self):
        points = [Point(1,1), Point(2, 3), Point(3, 1)]

        result = getDividedDifferences(points)
        self.assertTrue(numpy.allclose(result, [1, 2, -2]))

    def testInterpolate(self):
        points = [Point(-1, 0), Point(0, 1), Point(1, 0)]
        arguments = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

        result = interpolateNewton(points, arguments)
        self.assertTrue(numpy.allclose(result, [-15, -8, -3, 0, 1, 0, -3, -8, -15]))


if __name__ == "__main__":
    unittest.main()