from be_great import GReaT

import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

great = GReaT.load_from_dir("iris")

samples = great.sample(10, device="cpu", k=2)
print(samples)
