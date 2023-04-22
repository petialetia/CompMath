import numpy
import math
import random
import matplotlib.pyplot as plt

from GaussSeidelMethod import *

from sklearn.datasets import make_spd_matrix

def main():
    #means = []
    #dimensionalitys = [3, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500]

    dimensionality = 250

    coefficients_matrix = make_spd_matrix(dimensionality)

    constant_terms = numpy.array([random.random() for _ in range(dimensionality)])

    current_assumption = numpy.zeros(dimensionality)
    residual_logs = []

    for i in range(10):
        current_assumption = solveSLAEGaussSeidel(coefficients_matrix, constant_terms, \
                current_assumption, 1)

        residual_logs.append(numpy.linalg.norm(coefficients_matrix @ current_assumption - \
                constant_terms))
        print(residual_logs[i])

    fig, ax = plt.subplots()

    ax.plot(range(10), residual_logs )

    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("Невязка")

    plt.savefig('graphics.pdf')

    return 0

if __name__ == "__main__":
    main()