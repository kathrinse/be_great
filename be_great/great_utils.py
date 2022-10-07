import typing as tp

import numpy as np
import pandas as pd

from transformers import AutoTokenizer


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    """ Converts a Numpy Array to a Pandas DataFrame
    :param data: Pandas DataFrame or Numpy NDArray
    :param columns: If data is a Numpy Array, columns needs to be a list of all column names
    :return: Pandas DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(data[0]), \
        "%d column names are given, but array has %d columns!" % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """ Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}
    :param df: pandas DataFrame
    :param col: name of the column
    :return: distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_numpy()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist


def _convert_tokens_to_text(tokens: list[list[int]], tokenizer: AutoTokenizer):
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def _convert_text_to_tabular_data(text: list[str], df_gen: pd.DataFrame):
    columns = df_gen.columns.to_list()

    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns)

        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" is ")
            if values[0] in columns and not td[values[0]]:
                td[values[0]] = [values[1]]

        df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
    return df_gen
