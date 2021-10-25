from copy import deepcopy
from math import sqrt

import pandas


def read_csv_file(filename):
	return pandas.read_csv(filename)


def calculate_euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1[1]) - 1):
		distance += (row1[1][i] - row2[1][i]) ** 2
	return sqrt(distance)


def get_nearest_neighbors(train_data, test_row, k):
	distances = dict()

	for train in train_data:
		for train_row in train.iterrows():
			distances[calculate_euclidean_distance(train_row, test_row)] = train_row[1]["Type"]
	sorted_distances = sorted(distances.items(), key=lambda x: x[0])[:k]
	return [item[1] for item in sorted_distances]


def predict_type(nearest_neighbors_types):
	return max(set(nearest_neighbors_types), key=nearest_neighbors_types.count), nearest_neighbors_types


def normalize_dataset(dataset):
	normalized_dataset = dataset.copy()
	for column in list(dataset.columns.values):
		if column != "Type":
			normalized_dataset[column] = (normalized_dataset[column] - normalized_dataset[column].min()) / (
					normalized_dataset[column].max() - normalized_dataset[column].min())
	return normalized_dataset


def cross_validation_splitter(data, n_folds):
	dataframe_split = list()
	shuffle_data = data.sample(frac=1)
	for i in range(n_folds):
		dataframe_split.insert(i, fold_i_of_k(shuffle_data, i + 1, n_folds))
	return dataframe_split


def fold_i_of_k(dataset, i, k):
	n = len(dataset)
	return dataset[n * (i - 1) // k:n * i // k]


def accuracy_metric(original, predicted):
	correct = 0
	for i in range(len(original)):
		if original[i] == predicted[i][0]:
			correct += 1
	return correct / float(len(original)) * 100.0


def knn_algorithm(dataset, n_folds, k_neighbors):
	folds = cross_validation_splitter(dataset, n_folds)
	results = list()

	# create test and train sets
	for i in range(len(folds)):
		train_set = deepcopy(folds)
		test_set = train_set.pop(i)
		# original -> keeps original types for accuracy metric comparison
		original = list()

		for row in folds[i].iterrows():
			original.append(row[1]["Type"])

		predictions = list()
		for row in test_set.iterrows():
			output = predict_type(get_nearest_neighbors(train_set, row, k_neighbors))
			predictions.append(output)

		accuracy = accuracy_metric(original, predictions)
		results.append(accuracy)
	return results


def main():
	num_neighbors = [1, 3, 5, 7, 9]
	k_folds = 5
	data = read_csv_file("glass.csv.xls")
	# data_without_type = data.drop(columns=data.columns[-1], axis=1)
	normalized_data = normalize_dataset(data)

	for num in num_neighbors:
		scores = knn_algorithm(normalized_data, k_folds, num)
		print(scores)
		print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
		print("------------------------------------")


main()
