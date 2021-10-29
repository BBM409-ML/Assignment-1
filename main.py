from copy import deepcopy
from functools import reduce
from math import sqrt
import pandas


# converts given .xls file to pandas dataframe
def read_csv_file(filename):
	return pandas.read_csv(filename)


# calculates euclidean distance between two rows according to formula
def calculate_euclidean_distance(row1, row2):
	distance = 0.0
	# iterate over each column, calculate the distance, add it to total distance
	for i in range(len(row1[1]) - 1):
		distance += (row1[1][i] - row2[1][i]) ** 2
	return sqrt(distance)


# find the nearest neighbors of the test row
def get_nearest_neighbors(train_data, test_row, k):
	# distances are kept as dictionary because we need both type and distance
	# in order to sort distance and return the nearest ones' types
	distances = dict()

	for train in train_data:
		for train_row in train.iterrows():
			# calculate distance for each train row and test row
			distances[calculate_euclidean_distance(train_row, test_row)] = train_row[1]["Type"]
	# sort the dictionary values according to distances -> keys and get first k ones
	sorted_distances = sorted(distances.items(), key=lambda x: x[0])[:k]
	# return only the types
	return [item[1] for item in sorted_distances]


def predict_csMPa(train_data, test_row, k, target):
	distances = dict()

	for train in train_data:
		for train_row in train.iterrows():
			distances[calculate_euclidean_distance(train_row, test_row)] = train_row[1][target]
	sorted_distances = sorted(distances.items(), key=lambda x: x[0])[:k]
	return reduce(lambda a, b: a + b, [item[1] for item in sorted_distances]) / len(
		[item[1] for item in sorted_distances])


def predict_weighted_csMPa(train_data, test_row, k, target):
	distances = dict()

	for train in train_data:
		for train_row in train.iterrows():
			distances[calculate_euclidean_distance(train_row, test_row)] = train_row[1][target]
	sorted_distances = sorted(distances.items(), key=lambda x: x[0])[:k]

	x = reduce(lambda a, b: a + b,
			   [(1 / item[0] * item[1]) if item[0] != 0 else (item[1]) for item in sorted_distances])
	y = reduce(lambda a, b: a + b, [(1 / item[0]) if item[0] != 0 else (item[1]) for item in sorted_distances])

	return x / y


# returns the most occurred type value in the nearest neighbors
def predict_type(nearest_neighbors_types):
	return max(set(nearest_neighbors_types), key=nearest_neighbors_types.count)


# creates new normalized values by maintaining data distribution and scale
# normalization is done by considering min and max values in that particular column
def normalize_dataset(dataset, target):
	normalized_dataset = dataset.copy()
	for column in list(dataset.columns.values):
		if column != target:
			normalized_dataset[column] = (normalized_dataset[column] - normalized_dataset[column].min()) / (
					normalized_dataset[column].max() - normalized_dataset[column].min())
	return normalized_dataset


# splits dataframe into n folds
def cross_validation_splitter(data, n_folds):
	dataframe_split = list()
	# shuffle data
	shuffle_data = data.sample(frac=1)
	for i in range(n_folds):
		dataframe_split.insert(i, fold_i_of_k(shuffle_data, i + 1, n_folds))
	return dataframe_split


# returns ith fols
def fold_i_of_k(dataset, i, k):
	n = len(dataset)
	return dataset[n * (i - 1) // k:n * i // k]


# calculates the accuracy metric according to given formula
def accuracy_metric(original, predicted):
	correct = 0
	# compares predicted and original types
	for i in range(len(original)):
		if original[i] == predicted[i]:
			correct += 1
	return correct / float(len(original)) * 100.0


def mean_absolute_error(original, predicted):
	error = 0
	for i in range(len(original)):
		error += abs(original[i] - predicted[i])
	return error / len(original)


def predict_weighted_type(train_data, test_row, k):
	distances = dict()

	for train in train_data:
		for train_row in train.iterrows():
			distances[calculate_euclidean_distance(train_row, test_row)] = train_row[1]["Type"]
	sorted_distances = sorted(distances.items(), key=lambda x: x[0])[:k]

	weight = dict()
	for distance in sorted_distances:
		if weight.get(distance[1]) and distance[0] != 0:
			weight[distance[1]] = weight[distance[1]] + (1 / distance[0])
		elif (not weight.get(distance[1])) and distance[0] != 0:
			weight[distance[1]] = (1 / distance[0])
		elif distance[0] == 0:
			weight[distance[1]] = 0

	return max(weight, key=weight.get)


# cross validation, type prediction and accuracy calculation are done here
def knn_classification(dataset, n_folds, k_neighbors, algo_type, target):
	# create 5 folds from dataset
	folds = cross_validation_splitter(dataset, n_folds)
	results = list()

	# create test and train sets
	for i in range(len(folds)):
		# deepcopy data so that the original one does not change
		train_set = deepcopy(folds)
		# choose ith fold as test set and remove it from train data
		test_set = train_set.pop(i)
		# original -> keeps original types for accuracy metric comparison
		original = list()

		for row in folds[i].iterrows():
			original.append(row[1][target])

		predictions = list()
		# call the related function(weighted/non-weighted) for each test data with current train set
		# return the predicted types by the written algorithm
		for row in test_set.iterrows():
			if algo_type == "normal":
				output = predict_type(get_nearest_neighbors(train_set, row, k_neighbors))
				predictions.append(output)
			elif algo_type == "weighted":
				output = predict_weighted_type(train_set, row, k_neighbors)
				predictions.append(output)

		# calculate accuracy metric with original and predicted types
		accuracy = accuracy_metric(original, predictions)
		results.append(accuracy)

	return results


def knn_regression(dataset, n_folds, k_neighbors, algo_type, target):
	folds = cross_validation_splitter(dataset, n_folds)
	results = list()

	# create test and train sets
	for i in range(len(folds)):
		train_set = deepcopy(folds)
		test_set = train_set.pop(i)
		# original -> keeps original types for accuracy metric comparison
		original = list()

		for row in folds[i].iterrows():
			original.append(row[1][target])

		predictions = list()
		for row in test_set.iterrows():
			if algo_type == "normal":
				output = predict_csMPa(train_set, row, k_neighbors, target)
				predictions.append(output)
			elif algo_type == "weighted":
				output = predict_weighted_csMPa(train_set, row, k_neighbors, target)
				predictions.append(output)

		mea = mean_absolute_error(original, predictions)
		results.append(mea)
	return results


# this function is the main function for classification
def classification():
	# declares the main variables used across the functions
	num_neighbors = [1, 3, 5, 7, 9]
	k_folds = 5
	data = read_csv_file("glass.csv")
	target = "Type"
	normalized_data = normalize_dataset(data, target)

	# invokes KNN classification function by changing data and algorithm types
	print("KNN CLASSIFICATION WITHOUT FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("************************** K = " + str(num) + " ***********************************")
		scores = knn_classification(data, k_folds, num, "normal", target)

		for i in range(len(scores)):
			print("Fold {} Accuracy: {} ".format(i + 1, scores[i]))

		print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

	print()
	print("KNN CLASSIFICATION WITH FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("************************** K = " + str(num) + " ***********************************")
		scores = knn_classification(normalized_data, k_folds, num, "normal", target)

		for i in range(len(scores)):
			print("Fold {} Accuracy: {} ".format(i + 1, scores[i]))

		print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

	print()
	print("WEIGHTED KNN CLASSIFICATION WITHOUT FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("************************** K = " + str(num) + " ***********************************")
		scores = knn_classification(data, k_folds, num, "weighted", target)

		for i in range(len(scores)):
			print("Fold {} Accuracy: {} ".format(i + 1, scores[i]))

		print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

	print()
	print("WEIGHTED KNN CLASSIFICATION WITH FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("************************** K = " + str(num) + " ***********************************")
		scores = knn_classification(normalized_data, k_folds, num, "weighted", target)

		for i in range(len(scores)):
			print("Fold {} Accuracy: {} ".format(i + 1, scores[i]))

		print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


def regression():
	num_neighbors = [1, 3, 5, 7, 9]
	k_folds = 5
	data = read_csv_file("Concrete_Data_Yeh.csv")
	target = "csMPa"
	normalized_data = normalize_dataset(data, target)

	print()
	print("KNN REGRESSION WITHOUT FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("----------------------------- K = " + str(num) + " ----------------------------------------")
		scores = knn_regression(data, k_folds, num, "normal", target)

		for i in range(len(scores)):
			print("Fold {} MAE: {} ".format(i + 1, scores[i]))
		print('Mean Absolute Error (MAE): {}'.format(sum(scores) / float(len(scores))))

	print()
	print("KNN REGRESSION WITH FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("----------------------------- K = " + str(num) + " ----------------------------------------")
		scores = knn_regression(normalized_data, k_folds, num, "normal", target)

		for i in range(len(scores)):
			print("Fold {} MAE: {} ".format(i + 1, scores[i]))
		print('Mean Absolute Error (MAE): {}'.format(sum(scores) / float(len(scores))))

	print()
	print("WEIGHTED KNN REGRESSION WITHOUT FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("************************** K = " + str(num) + " ***********************************")
		scores = knn_regression(data, k_folds, num, "weighted", target)

		for i in range(len(scores)):
			print("Fold {} MAE: {} ".format(i + 1, scores[i]))
		print('Mean Absolute Error (MAE): {}'.format(sum(scores) / float(len(scores))))

	print()
	print("WEIGHTED KNN REGRESSION WITH FEATURE NORMALIZATION")
	for num in num_neighbors:
		print("************************** K = " + str(num) + " ***********************************")
		scores = knn_regression(normalized_data, k_folds, num, "weighted", target)

		for i in range(len(scores)):
			print("Fold {} MAE: {} ".format(i + 1, scores[i]))
		print('Mean Absolute Error (MAE): {}'.format(sum(scores) / float(len(scores))))


classification()
regression()
