import math as m
from feature import feature


def distance(x, y):
    # Simple Euclidean distance formula to get the distance between two points
    return m.sqrt(sum([m.pow(i - j, 2) for i, j in zip(x, y)]))


def get_feature_space(file):
    lines = [line.replace("\n", "").split(",") for line in open(file).readlines()]
    lines.pop(0)
    return [feature(line) for line in [[float(i) for i in line] for line in lines]]
