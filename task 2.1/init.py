import functions as f
import task_2_1 as t

k = 10

data_set = f.get_feature_space(f.knn_class, False)
sample = data_set[123]
knn = t.knn(sample, k, data_set)
estimate = t.classify(knn)
t.print_res(knn, estimate, sample, k, False)

data_set = f.get_feature_space(f.knn_reg, False)
sample = data_set[123]
knn = t.knn(sample, k, data_set)
estimate = t.regression(knn)
t.print_res(knn, estimate, sample, k, True)