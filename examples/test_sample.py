from be_great import GReaT

import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

great = GReaT.load_from_dir("iris_long")

samples = great.sample(150, device="cpu", k=50, start_col="target",
                       start_col_dist={"0.0": 0.33, "1.0": 0.33, "2.0": 0.33})
print(samples)

samples.to_csv("iris_long_samples.csv")
