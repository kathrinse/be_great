from be_great import GReaT
from sklearn import datasets

import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

df = datasets.load_iris(as_frame=True).frame
print(df.head())

great = GReaT("distilgpt2", epochs=10, save_steps=25, logging_steps=10)

trainer = great.fit(df)

great.save("iris")
