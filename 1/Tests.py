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

if __name__ == "__main__":
    unittest.main()