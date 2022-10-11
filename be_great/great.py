import os
import warnings
import json
import typing as tp
import logging

import numpy as np
import pandas as pd
import random

from tqdm import tqdm

import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments)

from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_start import GReaTStart, CategoricalStart, ContinuousStart, RandomStart
from be_great.great_trainer import GReaTTrainer
from be_great.great_utils import _array_to_dataframe, _get_column_distribution, _convert_tokens_to_text, \
    _convert_text_to_tabular_data


class GReaT:
    """ The GREAT pipeline.
        :param llm: HuggingFace Checkpoint to a pretrained large language model
        :param experiment_dir: Directory name where the training checkpoints will be saved
        :param epochs: Number of epochs to fine-tune the model
        :param batch_size: Batch size used for fine-tuning
        :param train_kwargs: TrainingArguments used by the HuggingFaceLibrary, see here the full list
                https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    """

    def __init__(self, llm: str, experiment_dir="trainer_great", epochs=100, batch_size=8, max_length=100,
                 **train_kwargs):
        # Load Model and Tokenizer from HuggingFace
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    def fit(self, data: tp.Union[pd.DataFrame, np.ndarray], column_names=None, conditional_col=None,
            resume_from_checkpoint=False) -> GReaTTrainer:
        """ Fine-tune a pretrained large language model to tabular data
            :param data: Pandas DataFrame or Numpy Array. Contains the tabular data
            :param column_names: List. If data is Numpy Array, the feature names have to be defined. If data is Pandas
            DataFrame, the value is ignored
            :param conditional_col: String. If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the last column is considered as conditional feature
            :param resume_from_checkpoint: If True, resumes training from the latest checkpoint in the experiment_dir.
            If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)
        """
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        training_args = TrainingArguments(self.experiment_dir,
                                          num_train_epochs=self.epochs,
                                          per_device_train_batch_size=self.batch_size,
                                          **self.train_hyperparameters)
        great_trainer = GReaTTrainer(self.model, training_args, train_dataset=great_ds, tokenizer=self.tokenizer,
                                     data_collator=GReaTDataCollator(self.tokenizer))

        # Start training
        logging.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def sample(self, n_samples: int, start_col="", start_col_dist=None,
               temperature=0.7, k=100, max_length=100, device="cuda") -> pd.DataFrame:
        """ Generate new synthetic samples
            :param n_samples: Number of samples to generate
            :param start_col: Feature to use as starting point for the generation process. If not given, the target
            learned during the fitting is used as starting point
            :param start_col_dist: Feature distribution of the starting feature. Should have the format
            "{F1: p1, F2: p2, ...}" for discrete columns or be a list of possible values for continuous columns.
            If not given, the target distribution learned during the fitting is used as starting point
            :param temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
            (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
            See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
            process
            :param k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
            :param max_length: Maximal number of tokens to generate - has to be long enough to not cut any information!
            :param device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU
        """
        great_start = self._get_start_sampler(start_col, start_col_dist)

        # Move model to device
        self.model.to(device)

        # Init empty DataFrame for the generated samples
        df_gen = pd.DataFrame(columns=self.columns)

        # Start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            while n_samples > df_gen.shape[0]:
                start_tokens = great_start.get_start_tokens(k)
                start_tokens = torch.tensor(start_tokens).to(device)

                # Generate tokens
                tokens = self.model.generate(input_ids=start_tokens, max_length=max_length,
                                             do_sample=True, temperature=temperature, pad_token_id=50256)

                # Convert tokens back to tabular data
                text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                df_gen = _convert_text_to_tabular_data(text_data, df_gen)

                # Remove rows with flawed numerical values
                for i_num_cols in self.num_cols:
                    df_gen = df_gen[pd.to_numeric(df_gen[i_num_cols], errors='coerce').notnull()]

                # Remove rows with missing values
                df_gen = df_gen.drop(df_gen[df_gen.isna().any(axis=1)].index)

                # Update process bar
                pbar.update(df_gen.shape[0] - already_generated)
                already_generated = df_gen.shape[0]

        df_gen = df_gen.reset_index(drop=True)
        return df_gen.head(n_samples)

    def great_sample(self, starting_prompts: tp.Union[str, list[str]], temperature=0.7, max_length=100, device="cuda"):
        """ Generate samples conditioned on an arbitrary input.
            :param starting_prompts: String or List of Strings on which the output is conditioned. For example,
            "Sex is female, Age is 26".
            :param temperature: The generation samples each token from the probability distribution given by a softmax
            function. The temperature parameter controls the softmax function. A low temperature makes it sharper
            (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
            See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
            process.
            :param max_length: Maximal number of tokens to generate - has to be long enough to not cut any information
            :param device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.
            
            ToDo: Set n_samples to generate more samples for one conditional input.
        """
        self.model.to(device)
        starting_prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        generated_data = []

        # Generate a sample for each starting point
        for prompt in tqdm(starting_prompts):
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)

            # Generate tokens
            gen = self.model.generate(input_ids=torch.unsqueeze(start_token, 0), max_length=max_length,
                                      do_sample=True, temperature=temperature, pad_token_id=50256)
            generated_data.append(torch.squeeze(gen))

        # Convert Text back to Tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        df_gen = _convert_text_to_tabular_data(decoded_data, pd.DataFrame(columns=self.columns))

        return df_gen

    def save(self, path: str):
        """ Save Model
            :param path: Directory to save model
        """
        # Make directory
        if os.path.isdir(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            # NDArray is not JSON serializable and therefore has to be converted into a list.
            if isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_finetuned_model(self, path: str):
        """ Load the weights of a fine-tuned large language model into the be_great pipeline
            :param path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        """ Load be_great class from directory
            :param path: Directory where model is saved
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_great model instance
        great = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        return great

    def _update_column_information(self, df):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(self, df, conditional_col=None):
        assert conditional_col is None or isinstance(conditional_col, str), \
            f"The column name has to be a string and not {type(conditional_col)}"
        assert conditional_col is None or conditional_col in df.columns, \
            f"The column name {conditional_col} is not in the feature names of the given dataset"

        # Take the distribution of the conditional column for a starting point in the generation process
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(self, start_col: tp.Optional[str],
                           start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]]) -> GReaTStart:
        if start_col and start_col_dist is None:
            raise ValueError(f"Start column {start_col} was given, but no corresponding distribution.")
        if start_col_dist is not None and not start_col:
            raise ValueError(f"Start column distribution {start_col} was given, the column name is missing.")

        assert start_col is None or isinstance(start_col, str), \
            f"The column name has to be a string and not {type(start_col)}"
        assert start_col_dist is None or isinstance(start_col_dist, dict) or isinstance(start_col_dist, list), \
            f"The distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}"

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)
