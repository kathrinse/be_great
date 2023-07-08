from be_great import GReaT
from sklearn import datasets

import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

data = datasets.load_iris(as_frame=True).frame
print(data.head())

column_names = ["sepal length", "sepal width", "petal length", "petal width", "target"]
data.columns = column_names

great = GReaT(
    "distilgpt2",
    epochs=50,
    save_steps=100,
    logging_steps=5,
    experiment_dir="trainer_iris",
    # lr_scheduler_type="constant", learning_rate=5e-5
)

trainer = great.fit(data, column_names=column_names)

great.save("iris")
