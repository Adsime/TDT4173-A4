import functions as f
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rfc
import task_2_3 as task

classifiers = [[knc(), "k Nearest  Neighbours Classifier"],
               [svm.SVC(gamma=0.001), "Support Vector Machine Classifier"],
               [rfc(), "Random Forest Classifier"]]

task.start(classifiers, f.get_digits(0.25))
