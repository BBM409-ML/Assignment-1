from math import sqrt
import pandas


def read_csv_file(filename):
	return pandas.read_csv(filename)


def calculate_euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)


def train_test_splitter(train_data):
	shuffle_df = train_data.sample(frac=1)
	train_size = int(0.7 * len(train_data))

	# Split the dataset
	train_set = shuffle_df[:train_size]
	test_set = shuffle_df[train_size:]

	return train_set, test_set


def get_nearest_neighbors(train_data, test_row, k):
	distances = dict()
	for i in range(len(train_data.index)):
		distances[calculate_euclidean_distance(train_data.iloc[i], test_row)] = train_data.iloc[i]["Type"]
	return sorted(distances.items(), key=lambda x: x[0])[:k]


def predict_type(nearest_neighbors_types):
	return max(set(nearest_neighbors_types), key=nearest_neighbors_types.count)


def normalize_dataset(dataset):
	normalized_dataset = dataset.copy()
	for column in list(dataset.columns.values):
		if column != "Type":
			normalized_dataset[column] = (normalized_dataset[column] - normalized_dataset[column].min()) / (
					normalized_dataset[column].max() - normalized_dataset[column].min())
	return normalized_dataset


def main():
	data = read_csv_file("glass.csv.xls")
	data_without_type = data.drop(columns=data.columns[-1], axis=1)
	print(normalize_dataset(data))
	train_set, test_set = train_test_splitter(data)
	neigh = get_nearest_neighbors(train_set, test_set.iloc[0], 5)


num_neighbors = [1, 3, 5, 7, 9]
k_folds = 5

main()
