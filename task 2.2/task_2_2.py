from sklearn import tree
import math as m
import matplotlib.pyplot as plt
import numpy as np


def train_classifier(data, weights, classifier:tree.DecisionTreeClassifier):
    x = [d.get_coords() for d in data]
    y = [d.get_target() for d in data]
    return classifier.fit(x, y, weights)


def find_error(c:tree.DecisionTreeClassifier, w, data):
    return 1 - c.score([x.get_coords() for x in data], [x.get_target() for x in data], w)


def get_zval(weight, a, prediction, target):
    return weight * m.exp(-a * target * prediction)


def train(data, init_weights, n_classifiers):
    classifiers = []
    alphas = []
    for i in range(n_classifiers):
        c = tree.DecisionTreeClassifier(max_depth=1)
        train_classifier(data, init_weights, c)
        classifiers.append(c)
        e = find_error(c, init_weights, data)
        a = 0.5 * m.log((1-e)/e)
        alphas.append(a)
        z = sum(get_zval(w, a, c.predict([x.get_coords()])[0], x.get_target()) for w, x in zip(init_weights, data))
        init_weights = [get_zval(w, a, c.predict([x.get_coords()])[0], x.get_target()) / z for w, x in zip(init_weights, data)]
    return [classifiers, alphas]


def test(data, classifiers, alphas):
    counter = 0
    res = [0] * len(data)
    for i, c in enumerate(classifiers):
        res += alphas[i] * c.predict([x.get_coords() for x in data])
    res = [1 if x > 0 else -1 for x in res]
    for r, x in zip(res, data):
        if r != x.get_target():
            counter += 1
    res = (counter/len(data))
    print("Error rate for " + len(classifiers).__str__() + " classifiers: " + res.__str__())
    return res


def plt_err(data):
    plt.plot(data[0], data[1])
    plt.xlabel("n classifiers")
    plt.ylabel("Error rate")
    plt.show()