from helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "data/dataset_to_release/dataset"
)

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test :", x_test.shape)
print("Premières étiquettes :", y_train[:10])

import numpy as np

print("NaN dans x_train :", np.isnan(x_train).sum())
print("NaN dans x_test :", np.isnan(x_test).sum())
