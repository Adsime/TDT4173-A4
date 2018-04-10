import functions as f
from sklearn import tree as t
import task_2_2 as task

classifier = t.DecisionTreeClassifier()
data_set = f.get_feature_space(f.databoost_train, True)
weights = [1 / x.get_index() for x in data_set]
test_set = f.get_feature_space(f.adaboost_test, True)

c, a = task.train(data_set, weights, 10)
res = [[], []]
for i in range(1, len(c) + 1):
    res[0].append(i)
    res[1].append(task.test(test_set, c[0:i], a[0:i]))
task.plt_err(res)