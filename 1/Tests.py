import unittest

from CholeskyDecomposition import *

class CholeskyDecompositionTestCase(unittest.TestCase):
    def testIdentityMatrix(self):
       coefficients_matrix = numpy.array([[1, 0], [0, 1]])
       constant_terms = [5, 6]

       self.assertTrue((factorizeMatrixCholesky(coefficients_matrix) == \
                numpy.linalg.cholesky(coefficients_matrix)).all())
       
       result = solveSLAECholesky(coefficients_matrix, constant_terms)
       self.assertTrue((result == [5, 6]).all())

    def testSimpltMatrix(self):
       coefficients_matrix = numpy.array([[2, 1], [1, 2]])
       constant_terms = [3, 4]

       self.assertTrue((factorizeMatrixCholesky(coefficients_matrix) == \
                numpy.linalg.cholesky(coefficients_matrix)).all())
       
       result = solveSLAECholesky(coefficients_matrix, constant_terms)
       self.assertTrue(numpy.allclose(result, [2 / 3, 5 / 3]))

if __name__ == "__main__":
    unittest.main()