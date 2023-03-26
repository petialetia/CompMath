import numpy
import itertools

def main():
    print(solveSLAEGaussSeidel(numpy.array([[1000, 1, 1], [1, 1000, 1], [1, 1, 1000]]), \
            [5013, 6012, 7011]))

    return 0

def solveSLAEGaussSeidel(coefficients_matrix, constant_terms, start_assumption = None, \
        max_iteration = None, target_residual = 1e-08):
    dimensionality = len(constant_terms)

    if (start_assumption is None):
        start_assumption = numpy.zeros(dimensionality)

    current_answer = start_assumption

    for _ in itertools.count() if max_iteration is None else range(max_iteration):
        next_step_answer = numpy.zeros(dimensionality)

        for i in range(len(constant_terms)):
            next_step_answer[i] = (constant_terms[i] - \
                    sum(coefficients_matrix[i][j] * next_step_answer[j] for j in range(i)) - \
                    sum(coefficients_matrix[i][j] * current_answer[j] for j in range( \
                    i + 1, dimensionality))) / coefficients_matrix[i][i]
            
        current_answer = next_step_answer

        if (numpy.allclose(coefficients_matrix @ current_answer, constant_terms, \
                atol = target_residual)):
            break

    return current_answer

if __name__ == "__main__":
    main()