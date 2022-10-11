import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """
    The GReaTDataset overwrites the _getitem function to include the permutation.
    """
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """
        Get one instance of the tabular data, permuted, converted to text and tokenized
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)

        shuffled_text = ", ".join(
            ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx]
        )

        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """
    Overwrites the DataCollator to also pad the labels and not only the input ids
    """
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
