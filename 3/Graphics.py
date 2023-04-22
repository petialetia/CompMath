import numpy
import math
import random
import matplotlib.pyplot as plt

from NewtonInterpolation import *

def main():
    runge_function = lambda x: 1/(1 + 25 * x**2)

    left_border = -1
    right_border = 1

    drawInterpolation(runge_function, \
            getPointsUniform(runge_function, left_border, right_border, 20), \
            getArgumentsUniform(left_border, right_border, 200), "Uniform.pdf")

    return 0

def getArgumentsUniform(left_border, right_border, num):
    arguments = []

    step = (right_border - left_border) / (num - 1)

    current_argument = left_border

    for _ in range(num):
        arguments.append(current_argument)
        current_argument += step

    return arguments

def getPointsUniform(function, left_border, right_border, num):
    return [Point(argument, function(argument)) for argument in 
            getArgumentsUniform(left_border, right_border, num)]

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