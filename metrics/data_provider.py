import pandas as pd
import os
import warnings
import json
import numpy as np

class DataProvider():
    """ An interface that allows accessing original and generated data for any dataset. """

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, gen_models: dict):
        """ Initialize the data provider.
            :param train_data: The original training dataset
            :param test_data: The original testing dataset
            :param gen_models: trained generative model that should be evaluated, e.g. GReaT.
                Each model is an value in a dictionary, with a model name as key.
            :param dataset_config: A dictionary with meta information on the dataset.
        """
        self.orig_data_train = train_data
        self.orig_data_test = test_data
        self.mymodels = gen_models

        ## generate the datasets.
        self.gen_test_datasets = {} # dict of generated "train" datasets 
        self.gen_train_datasets = {} # dict of generated "test" datasets
        for key, model in self.mymodels.items():
            print("Generating data for model", key, "...")
            self.gen_train_datasets[key] = model.sample(n_samples=len(self.orig_data_train))
            self.gen_test_datasets[key] = model.sample(n_samples=len(self.orig_data_test))

    def get_full_data(self, model_type: str, train=True):
        """ Return the full dataset or a generated dataset the has the same shape as the original dataset.
            :param dataset_name: Name of dataset. Currently supports {adult, california, heloc, travel}
            :param model_type: Model used to generate data (the key in gen_models passed in constructor) or "original". 
            :param train: Return train or test set (only for model_type=original, otherwise only the dataset size is adaped)
            :param hyperparams: Other model hyperparameters, such as temperature for distill, number of neighbors for smote...
            returns a pandas.DataFrame of the data.
        """
        if model_type == "original":
            if train:
                df = self.orig_data_train
            else:
                df = self.orig_data_test
        else:
            if train:
                df = self.gen_train_datasets[model_type]
            else:
                df = self.gen_test_datasets[model_type]

        if df.isnull().any().any(): # contains nan
            old_len = len(df)
            df = df.dropna()
            print("Warning: Dropped ", old_len-len(df), "records because of NaN values.")
        return df
    
    def get_random_data_sample(self, model_type: str, n_samples = 100, train=True):
        """ Return a random sample of the dataset.
            :param dataset_name: Name of dataset. Currently supports {adult, california, heloc, travel, diabetes}
            :param model_type: Model used to generate data or "original". Currently supports {original, tvae, ctgan, distillgpt2}
            :param train: Return train or test set (only for model_type=original) ##TODO: this needs to be adjusted for all data types
            :param n_samples: number of samples to draw with replacement.
            returns a pandas.DataFrame of the data.
        """
        if model_type == "original":
            if train:
                df = self.orig_data_train
            else:
                df = self.orig_data_test
        else:
            if train:
                df = self.gen_train_datasets[model_type]
            else:
                df = self.gen_test_datasets[model_type]

        return df.sample(n=n_samples, replace = True)

    def get_config(self, dataset_name: str) -> dict:
        """ Return a dict containing information about the dataset.
            :param dataset_name: Name of dataset. Currently supports {adult, california, heloc, travel, diabetes, midwest}
        """ 
        with open(f"metrics/config/{dataset_name}/config.json", "r") as f:
            config = json.load(f)
            
        return config
