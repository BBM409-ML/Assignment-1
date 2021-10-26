from copy import deepcopy
from functools import reduce
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


def predict_csMPa(train_data, test_row, k, target):
    distances = dict()

    for train in train_data:
        for train_row in train.iterrows():
            distances[calculate_euclidean_distance(train_row, test_row)] = train_row[1][target]
    sorted_distances = sorted(distances.items(), key=lambda x: x[0])[:k]
    return reduce(lambda a, b: a + b, [item[1] for item in sorted_distances]) / len([item[1] for item in sorted_distances])


def predict_weighted_csMPa(train_data, test_row, k, target):
    distances = dict()

    for train in train_data:
        for train_row in train.iterrows():
            distances[calculate_euclidean_distance(train_row, test_row)] = train_row[1][target]
    sorted_distances = sorted(distances.items(), key=lambda x: x[0])[:k]

    x = reduce(lambda a, b: a + b, [(1/item[0]*item[1]) if item[0] != 0 else (item[1]) for item in sorted_distances])
    y = reduce(lambda a, b: a + b, [(1/item[0]) if item[0] != 0 else (item[1]) for item in sorted_distances])

    return x/y


def predict_type(nearest_neighbors_types):
    return max(set(nearest_neighbors_types), key=nearest_neighbors_types.count)


def normalize_dataset(dataset, target):
    normalized_dataset = dataset.copy()
    for column in list(dataset.columns.values):
        if column != target:
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


def knn_classification(dataset, n_folds, k_neighbors, algo_type, target):
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
                output = predict_type(get_nearest_neighbors(train_set, row, k_neighbors))
                predictions.append(output)
            elif algo_type == "weighted":
                output = predict_weighted_type(train_set, row, k_neighbors)
                predictions.append(output)

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


def classification():
    num_neighbors = [1, 3, 5, 7, 9]
    k_folds = 5
    data = read_csv_file("../../Desktop/glass.csv.xls")
    target = "Type"
    normalized_data = normalize_dataset(data, target)

    print("KNN CLASSIFICATION WITHOUT FEATURE NORMALIZATION")
    for num in num_neighbors:
        print("************************** K = "+str(num)+" ***********************************")
        scores = knn_classification(data, k_folds, num, "normal", target)

        for i in range(len(scores)):
            print("Fold {} Accuracy: {} ".format(i+1, scores[i]))

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
        print("----------------------------- K = "+str(num)+" ----------------------------------------")
        scores = knn_regression(data, k_folds, num, "normal", target)

        for i in range(len(scores)):
            print("Fold {} MAE: {} ".format(i+1, scores[i]))
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
            print("Fold {} MAE: {} ".format(i+1, scores[i]))
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
