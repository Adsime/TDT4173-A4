import math as m


def distance(x, y):
    """
    Simple Euclidean distance formula to get the distance between two points
    :param x: array
    :param y: array
    :return: float
    """
    dist = 0
    for i, j in zip(x, y):
        dist += m.pow(i - j, 2)
    return m.sqrt(dist)


def read_file(file):
    lines = []
    for line in open(file).readlines():
        lines.append(line.replace("\n", "").split(","))
    lines.pop(0)
    return [[float(i) for i in line] for line in lines]
