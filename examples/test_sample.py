from be_great import GReaT

import logging
from utils import set_logging_level
from sklearn import datasets

logger = set_logging_level(logging.INFO)

great = GReaT.load_from_dir("iris")

# Continuous column as start
# data, target = datasets.load_iris(return_X_y=True)
# sepal = list(data[:, 0])
# samples = great.sample(20, device="cpu", k=5, start_col="sepal length", start_col_dist=sepal)

# Random Start
# samples = great.sample(12, device="cpu", k=6)

# Categorical column as start
samples = great.sample(
    20, k=5, start_col="target", start_col_dist={"0.0": 0.33, "1.0": 0.33, "2.0": 0.33}
)

print(samples)
samples.to_csv("iris_samples.csv")
