import numpy
import math

def main():
    print(solveSLAECholesky(numpy.matrix('1 0; 0 1'), [5, 6]))

    return 0

def solveSLAELowerTriangular(coefficients_matrix, constant_terms):
    result = numpy.zeroes(constant_terms.size())
    
    for i in range(constant_terms.size()):
        sum = 0

        for j in range(i):
            sum += result[j] * coefficients_matrix[i][j]

        result[i] = (constant_terms[i] - sum) / coefficients_matrix[i][i]

    return result

def solveSLAEUpperLowerTriangular(coefficients_matrix, constant_terms):
    result = numpy.zeroes(constant_terms.size())
    
    for i in (range(constant_terms.size())).reversed():
        sum = 0

        for j in range(i, constant_terms.size()):
            sum += result[j] * coefficients_matrix[i][j]

        result[i] = (constant_terms[i] - sum) / coefficients_matrix[i][i]

    return result

def solveSLAECholesky(coefficients_matrix, constant_terms):
    factorization = factorizeMatrixCholesky(coefficients_matrix)

    itermidiate_constant_terms = solveSLAELowerTriangular(factorization, constant_terms)

    return solveSLAEUpperLowerTriangular(factorization.transpose(), itermidiate_constant_terms)


def factorizeMatrixCholesky(matrix):
    assert(len(matrix.shape) == 2)
    assert(matrix.shape[0] == matrix.shape[1])

    matrix_dimensionality = (matrix.shape)[0]

    factorization = numpy.zeros(matrix.shape)

    for i in range(matrix_dimensionality):
        sum = 0

        for j in range(i):
            sum += factorization[j][j] ** 2

        factorization[i][i] = math.sqrt(matrix[i][i] - sum)

    for i in range(matrix_dimensionality):
        for j in range(i):
            sum = 0

            for k in range(j):
                sum += factorization[i][k] * factorization[j][k]

            factorization[i][j] = (matrix[i][j] - sum) / factorization[j][j]

    return factorization

if __name__ == "__main__":
    main()