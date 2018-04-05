import functions as f
import task as t
from feature import feature as ft

knn_class = "./../data/knn_classification.csv"
knn_reg = "./../data/knn_regression.csv"

k = 10

data_set = [ft(data) for data in f.read_file(knn_class)]
sample = data_set.pop(123)
knn = t.knn(sample, k, data_set)
estimate = t.classify(knn)
t.print_res(knn, estimate, sample, k, False)

data_set = [ft(data) for data in f.read_file(knn_reg)]
sample = data_set.pop(123)
knn = t.knn(sample, k, data_set)
estimate = t.reg(knn)
t.print_res(knn, estimate, sample, k, True)