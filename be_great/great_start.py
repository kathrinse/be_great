import random
import numpy as np


def pad(x, length, pad_value=50256):
    return [pad_value] * (length - len(x)) + x


# Check that all lists of tokens have the same length, pad them if necessary to the max length
def _pad_tokens(tokens):
    max_length = len(max(tokens, key=len))
    tokens = [pad(t, max_length) for t in tokens]
    return tokens


class GReaTStart:
    """ Abstract super class to handle the creation of tokens to start the generation process.
        :param tokenizer: Tokenizer used to tokenize the input string
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_start_tokens(self, n_samples):
        """
        Creates starting points for the generation process
        :param n_samples: Number of start prompts to create
        :return: list of n_samples lists of tokens
        """
        raise NotImplementedError("This has to be overwritten but the subclasses")


class CategoricalStart(GReaTStart):
    """
    For a categorical column as starting point:
    Uses the categories of the specific column as start point.
    """
    def __init__(self, tokenizer, start_col, start_col_dist):
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, dict), ""

        self.start_col = start_col
        self.population = list(start_col_dist.keys())
        self.weights = list(start_col_dist.values())

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.population, self.weights, k=n_samples)
        start_text = [self.start_col + " is " + str(s) + "," for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text)["input_ids"])
        return start_tokens


class ContinuousStart(GReaTStart):
    """
    For a continuous column as starting point:
    Uses the values (with random noise) of the specific column as start point.
    """
    def __init__(self, tokenizer, start_col, start_col_dist, noise=.01, decimal_places=5):
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, list), ""

        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.noise = noise
        self.decimal_places = decimal_places

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.start_col_dist, k=n_samples)
        start_words += np.random.normal(size=n_samples) * self.noise  # add noise to start words
        start_text = [self.start_col + " is " + format(s, f".{self.decimal_places}f") + "," for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text)["input_ids"])
        return start_tokens


class RandomStart(GReaTStart):
    """
    Can be used if no distribution of any column is known.
    Just uses different column names as starting point.
    """
    def __init__(self, tokenizer, all_columns):
        super().__init__(tokenizer)
        self.all_columns = all_columns

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.all_columns, k=n_samples)
        start_text = [s + " is " for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text)["input_ids"])
        return start_tokens
