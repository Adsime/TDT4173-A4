import math as m
from feature import feature
import sklearn.datasets as data
import sklearn.model_selection as test
import matplotlib.pyplot as plt


knn_class = "./../data/knn_classification.csv"
knn_reg = "./../data/knn_regression.csv"
adaboost_test = "./../data/adaboost_test.csv"
databoost_train = "./../data/adaboost_train.csv"


def distance(x, y):
    # Simple Euclidean distance formula to get the distance between two points
    return m.sqrt(sum([m.pow(i - j, 2) for i, j in zip(x, y)]))


def get_feature_space(file, is_adaboost):
    lines = [line.replace("\n", "").split(",") for line in open(file).readlines()]
    lines.pop(0)
    # Converts each item in the matrix 'lines' to floats as well as generating feature objects of each row.
    return [feature(line, is_adaboost) for line in [[float(i) for i in line] for line in lines]]


def get_digits(split):
    d = data.load_digits()
    return test.train_test_split(d.images, d.target, test_size=split, random_state=10)


def show_num(image):
    plt.gray()
    plt.matshow(image)
    plt.show()