import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix as cm


def start(classifiers, data):
    # Split the data
    X_train, Y_test, x_train, y_test = data

    # Flatten the arrays to be 2-dim instead of 3-dim as the fit method doesn't accept 3-dim data
    X_train = [np.ravel(x) for x in X_train]
    Y_test = [np.ravel(x) for x in Y_test]

    # Train the classifiers on the training data
    for x, y in classifiers:
        x.fit(X_train, x_train)
    # Build a confusion matrix for each classifier
    cms = [(cm(y_test, x.predict(Y_test)), y) for x, y in classifiers]

    plot_confusion_matrix(cms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def plot_confusion_matrix(cms, classes,
                          normalize=False,
                          title=' Confusion matrix',
                          cmap=plt.cm.Blues):

    # Loop over each classifier and
    for x, y in cms:
        plt.imshow(x, interpolation='nearest', cmap=cmap)
        plt.title(y + title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = x.max() / 2.
        for i, j in itertools.product(range(x.shape[0]), range(x.shape[1])):
            plt.text(j, i, format(x[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if x[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Target')
        plt.xlabel('Prediction')
        plt.figure()
    plt.show()