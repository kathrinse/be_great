from be_great import GReaT
from sklearn import datasets
import numpy as np

import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

# data = datasets.load_iris(as_frame=True).frame
# print(data.head())

data, target = datasets.load_iris(return_X_y=True)
# target = target.astype("object")
# target[target == 0] = "Iris-Setosa"
# target[target == 1] = "Iris-Versicolor"
# target[target == 2] = "Iris-Virginica"
data = np.concatenate((data, np.array(target).reshape(-1, 1)), axis=1)
column_names = ["sepal length", "sepal width", "petal length", "petal width", "target"]

great = GReaT("distilgpt2", epochs=50, save_steps=100, logging_steps=25, experiment_dir="trainer_iris_long")

trainer = great.fit(data, column_names=column_names)

great.save("iris_long")
