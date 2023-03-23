import unittest

from GaussSeidelMethod import *

class CholeskyDecompositionTestCase(unittest.TestCase):
    def testIdentityMatrix(self):
        coefficients_matrix = numpy.array([[1, 0], [0, 1]])
        constant_terms = [5, 6]
        target_residual = 1e-05

        result = solveSLAEGaussSeidel(coefficients_matrix, constant_terms, target_residual = target_residual)
        self.assertTrue(numpy.allclose(coefficients_matrix @ result, constant_terms, target_residual))

    def testSimpleMatrix(self):
        coefficients_matrix = numpy.array([[2, 1], [1, 2]])
        constant_terms = [3, 4]
        target_residual = 1e-05

        result = solveSLAEGaussSeidel(coefficients_matrix, constant_terms, target_residual = target_residual)
        self.assertTrue(numpy.allclose(coefficients_matrix @ result, constant_terms, target_residual))

    def testComplicatedMatrix(self):
        coefficients_matrix = numpy.array([[1000, 1, 1], [1, 1000, 1], [1, 1, 1000]])
        constant_terms = [5013, 6012, 7011]
        target_residual = 1e-05
       
        result = solveSLAEGaussSeidel(coefficients_matrix, constant_terms, target_residual = target_residual)
        self.assertTrue(numpy.allclose(coefficients_matrix @ result, constant_terms, target_residual))


if __name__ == "__main__":
    unittest.main()