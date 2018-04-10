import functions as f
from collections import defaultdict
import random


def knn(sample, k, data_set):
    # Places the Euclidean distance of each set to the sample in an array
    distances = [f.distance(feature.get_coords(), sample.get_coords()) for feature in data_set]
    res = distances.copy()
    # Sorting to have easy access to the shortest distances
    res.sort()
    # Return the features that are of shortest distance to the sample
    return [data_set[distances.index(res[i])] for i in range(0, k)]


def classify(features):
    classes = defaultdict(int)
    # Count occurrences of each class in the feature set
    for f in features:
        classes[f.get_target()] += 1
    winners = []
    # This loop is constructed to avoid same class always being chosen if there are equal contestants.
    for c in classes:
        if len(winners) == 0 or classes[winners[0]] == classes[c]:
            winners.append(c)
        elif classes[winners[0]] < classes[c]:
            winners[0] = c
    # Chooses a random winner
    return random.choice(winners)


def regression(features):
    # Mean value
    return sum([f.get_target() for f in features]) / len(features)


def print_res(knn, estimate, sample, k, regression):
    print("----- REGRESSION -----" if regression else "----- CLASSIFICATION -----")
    print(k.__str__() + " nearest neighbours")
    print([n.get_coords() for n in knn])
    print("Result for sample: " + sample.get_coords().__str__() + ": " + estimate.__str__() + "\n\n")
