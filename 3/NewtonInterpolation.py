class Point:
    argument: float
    value: float

    def __init__(self, argument: float, value: float):
        self.argument = argument
        self.value = value

def main():
    # points = [Point(1,1), Point(2, 3), Point(3, 1)]
    # print(getDividedDifferences(points))

    print(interpolateNewton([Point(-1, 0), Point(0, 1), Point(1, 0)], [-3, -2, -1, 0, 1, 2, 3]))

def interpolateNewton(points, arguments):
    divided_differences = getDividedDifferences(points)

    values = []

    for argument in arguments:
        value = divided_differences[0]
        koef = 1

        for i in range(1, len(points)):
            koef *= (argument - points[i-1].argument)
            value += koef * divided_differences[i]

        values.append(value)

    return values

def getDividedDifferences(points):
    divided_differences = []
    
    for _ in range(len(points)):
        divided_differences.append([])

    for i in range(len(points)):
        divided_differences[0].append(points[i].value)

    for i in range(1, len(points)):
        for j in range(len(points) - i):
            divided_differences[i].append( \
                    (divided_differences[i-1][j+1] - divided_differences[i-1][j]) / \
                    (points[j + i].argument - points[j].argument))

    return [row[0] for row in divided_differences]

if __name__ == "__main__":
    main()