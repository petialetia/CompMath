import numpy
import time
import random
import matplotlib.pyplot as plt

from CholeskyDecomposition import *

def main():
    means = []
    dimensionalitys = [3, 10, 50, 100, 250, 500, 750, 1000]

    for dimensionality in dimensionalitys:
        time = []

        coefficients_matrix = numpy.diag(numpy.full(dimensionality,1))
        constant_terms = numpy.array([random.random() for _ in range(dimensionality)])
        time.append(getTimeOfSolving(coefficients_matrix, constant_terms))

        print("first {}".format(dimensionality))

        coefficients_matrix = numpy.diag(numpy.full(dimensionality,10 ** 3)) + \
                numpy.diag(numpy.ones(dimensionality - 1), 1) + \
                numpy.diag(numpy.ones(dimensionality - 1), -1)
        constant_terms = numpy.array([random.random() for _ in range(dimensionality)])
        time.append(getTimeOfSolving(coefficients_matrix, constant_terms))

        print("second {}".format(dimensionality))

        coefficients_matrix = numpy.zeros((dimensionality, dimensionality), int)
        coefficients_matrix[range(dimensionality), range(dimensionality)] = 10 ** 3
        coefficients_matrix[range(dimensionality - 2), range(2, dimensionality)] = 1
        coefficients_matrix[range(2, dimensionality), range(dimensionality - 2)] = 1
        constant_terms = numpy.array([random.random() for _ in range(dimensionality)])
        time.append(getTimeOfSolving(coefficients_matrix, constant_terms))

        print("third {}".format(dimensionality))

        coefficients_matrix = numpy.zeros((dimensionality, dimensionality), int)
        coefficients_matrix[range(dimensionality), range(dimensionality)] = 10 ** 5
        coefficients_matrix[range(dimensionality - 2), range(2, dimensionality)] = 10
        coefficients_matrix[range(2, dimensionality), range(dimensionality - 2)] = 10
        coefficients_matrix[range(dimensionality - 1), range(1, dimensionality)] = 1
        coefficients_matrix[range(1, dimensionality), range(dimensionality - 1)] = 1
        constant_terms = numpy.array([random.random() for _ in range(dimensionality)])
        time.append(getTimeOfSolving(coefficients_matrix, constant_terms))

        print("fouth {}".format(dimensionality))

        means.append(numpy.mean(time))

    fig, ax = plt.subplots()

    ax.plot(dimensionalitys, means)

    ax.set_xlabel("Размерность")
    ax.set_ylabel("Среднее время подсчёта")

    """plt.legend(loc="best")

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(range(0, 9, 1))"""

    plt.savefig('graphics.pdf')

    return 0

def getTimeOfSolving(coefficients_matrix, constant_terms):
    start_time = time.time()
    solveSLAECholesky(coefficients_matrix, constant_terms)
    finish_time = time.time()

    return finish_time - start_time

if __name__ == "__main__":
    main()