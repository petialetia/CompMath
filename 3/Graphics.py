import numpy
import math
import random
import matplotlib.pyplot as plt

from NewtonInterpolation import *

def main():
    runge_function = lambda x: 1/(1 + 25 * x**2)

    left_border = -1
    right_border = 1
    n_interpolation_nodes = 15
    n_graphic_points = 200

    drawInterpolation(runge_function, \
            getPoints(runge_function, left_border, right_border, \
            n_interpolation_nodes, getArgumentsUniform), \
            getArgumentsUniform(left_border, right_border, n_graphic_points), "Uniform.pdf")

    drawInterpolation(runge_function, \
            getPoints(runge_function, left_border, right_border, \
            n_interpolation_nodes, getArgumentsChebyshev), \
            getArgumentsChebyshev(left_border, right_border, n_graphic_points), "Chebyshev.pdf")

    return 0

def getArgumentsUniform(left_border, right_border, num):
    arguments = []

    step = (right_border - left_border) / (num - 1)

    current_argument = left_border

    for _ in range(num):
        arguments.append(current_argument)
        current_argument += step

    return arguments

def getArgumentsChebyshev(left_border, right_border, num):
    return [(left_border + right_border) / 2 + \
            (right_border - left_border) * math.cos((2 * i + 1) * math.pi / (2 * num)) / 2 \
            for i in range(num)]

def getPoints(function, left_border, right_border, num, get_arguments):
    return [Point(argument, function(argument)) for argument in 
            get_arguments(left_border, right_border, num)]



def drawInterpolation(function, nodes_of_interpolation, arguments, file_name):
    values = interpolateNewton(nodes_of_interpolation, arguments)

    fig, ax = plt.subplots()

    ax.plot(arguments, [function(argument) for argument in arguments], \
            linestyle = "-", color = "r")
    ax.plot(arguments, values, linestyle = "--", color = "b")
    ax.plot([point.argument for point in nodes_of_interpolation], 
            [point.value for point in nodes_of_interpolation], 
            marker = "o", linestyle = "", color = "b")

    plt.savefig(file_name)

if __name__ == "__main__":
    main()