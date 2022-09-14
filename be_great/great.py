import os
import warnings
import json
import typing as tp

import numpy as np
import pandas as pd
import random

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          TrainingArguments, 
                          Trainer)
from datasets import Dataset


class GREAT:
    
    """ The GREAT method.
        :param llm: HuggingFace Checkpoint to a pretrained large language model
        :param experiment_name: Directory name where the training checkpoints will be saved
        :param epochs: Number of epochs to fine-tune the model
        :param batch_size: Batch size used for fine-tuning
        :param train_params: TrainingArguments used by the HuggingFaceLibrary, see here the full list https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    """

    def __init__(self, llm: str, experiment_name="trainer_great", epochs=100, batch_size=8, **train_params):
        self.llm = llm
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.llm)
        
        self.experiment_name = experiment_name
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_params = train_params
        
        self.target_col = None
        
    def fit(self, df: pd.DataFrame, target_col=None, verbose=False, resume_from_checkpoint=False):
        """ Fine-tune a pretrained large language model to tabular data.
            :param df: Pandas DataFrame containing tabular data
            :param target_col: If given, the distribution is saved and used as a starting point for the generation process later. If not, the last feature is considered as target feature.
        """
        
        # Save the column names
        self.columns = df.columns.to_list()
        
        # Save names of numerical columns for some sanity checks later
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()
        
        # Save the distribution of the target column as a starting point for the generation process
        self.target_col = target_col if target_col else df.columns[-1]
        
        if df[self.target_col].dtype == "float":
            self.target_col_dist = df[self.target_col].to_numpy()
        else:
            self.target_col_dist = df[self.target_col].value_counts(1).to_dict()
        
        if verbose:
            print("Convert data into HuggingFace dataset object")
                    
        # Convert DataFrame into HuggingFace dataset object
        ds = Dataset.from_pandas(df)
        
        # Define function to convert the tabular data into text data and shuffle it
        def combine_data_shuffled(sample):
            concat = ""
            for col in random.sample(self.columns, k=len(self.columns)):
                concat += "%s is %s, " % (col, str(sample[col]).strip())
            return {"concat": concat}
        
        # Combine tabular data together into text data
        combined_ds = ds.map(combine_data_shuffled)
        
        # Remove all original features from the dataset 
        combined_ds = combined_ds.remove_columns(ds.column_names)
        
        if verbose:
            print("Textualized tabular data")
            print(combined_ds["concat"][:3])
        
        # We have to make this ugly max length padding, because the usual dynamic padding does not work on the labels features, but we still need it, as the rows consist of different numbers of tokens. Otherwise the training will fail. The max_length value should be set to a value, that is close to maximum number of tokens one row will need. If it is too low, the train data is truncated. If it is too high, this leads to an unnecessary overhead.
        
        # Estimate longest sequence length - Is there a way to make this better?
        example_ids = self.tokenizer(combined_ds["concat"][:100])["input_ids"]
        self.max_length = len(max(example_ids, key=len)) + 10
        
        def tokenizer_function(sample):
            result = self.tokenizer(sample["concat"], truncation=True, padding="max_length", max_length=self.max_length)
            result["labels"] = result["input_ids"].copy()
            return result
        
        if verbose:
            print("Tokenize data...")
        
        # Tokenize dataset and create pytorch tensors
        tokenizer_ds = combined_ds.map(tokenizer_function, batched=True)
        tokenizer_ds.set_format("torch")
        
        # Set training parameters
        training_args = TrainingArguments(self.experiment_name,
                                          num_train_epochs=self.epochs,
                                          per_device_train_batch_size=self.batch_size,
                                          **self.train_params)
        
        if verbose:
            print("Start training...")
            
        trainer = Trainer(self.model, training_args, train_dataset=tokenizer_ds, tokenizer=self.tokenizer)
        
        # Start training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        if verbose:
            rounds = len(trainer.state.log_history) - 1
            loss = [x["loss"] if i < rounds else None for i, x in enumerate(trainer.state.log_history)]
            plt.plot(loss)
            
    
    def sample(self, n_samples: int, start_col="", start_col_dist={}, temperature=0.7, k=100, device="cuda"):
        """ Generate new synthetic samples
            :param n_samples: Number of samples to generate
            :param start_col: Feature to use as starting point for the generation process. If not given, the target learned during the fitting is used as starting point.
            :param start_col_dist: Feature distribution of the starting feature. Should have the format "{F1: p1, F2: p2, ...}" for discrete columns or be a list of possible values for continuous columns. If not given, the target distribution learned during the fitting is used as starting point.
            :param temperature: The generation samples each token from the probility distribution given by a softmax function. The temperature parameter controlls the softmax function. A low temperature makes it sharper (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output. See this blog articel (https://huggingface.co/blog/how-to-generate) to read more about the generation process.
            :param k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly.
            :param device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.
        """        
        if start_col and not start_col_dist:
            raise ValueError(f"Start column {start_col} was given, but no corresponding distribution.")
        if start_col_dist and not start_col:
            raise ValueError(f"Start column distribution {start_col} was given, the column name is missing.")
        
        start_col = start_col if start_col else self.target_col
        start_col_dist = start_col_dist if start_col_dist else self.target_col_dist
        
        assert start_col, f"There is no learned target distribution.\
                            This may occur, when you did not fine-tune the model via the fitting function. \
                            Therefore, to sample data, you have to specify a start column and a start column distribution."
        
        col_type = "discrete" if isinstance(start_col_dist, dict) else "continuous"
            
        self.model.to(device)
        
        df_gen = pd.DataFrame(columns=self.columns)
        
        
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            while n_samples > df_gen.shape[0]:

                # Create start prompt
                if col_type == "discrete":
                    population = list(start_col_dist.keys())
                    weights = list(start_col_dist.values())

                    start_words = random.choices(population, weights, k=k)
                    start = [start_col + " is " + str(s) + "," for s in start_words]
                else:
                    start_words = random.choices(start_col_dist, k=k)
                    start_words += np.random.normal(size=k)*.01 # add noise to start words
                    start = [start_col + " is "+ format(s, ".5f") + "," for s in start_words]

                start_token = torch.tensor(self.tokenizer(start)["input_ids"]).to(device)

                # Generate tokens
                tokens = self.model.generate(input_ids=start_token, max_length=self.max_length+20,
                                     do_sample=True, temperature=temperature, pad_token_id=50256)

                
                # Convert tokens back to tabular data
                text_data = convert_tokens_to_text(tokens, self.tokenizer)
                df_gen = convert_text_to_tabular_data(text_data, df_gen)

                # Remove rows with flawed numerical values
                for i_num_cols in self.num_cols:
                    df_gen = df_gen[pd.to_numeric(df_gen[i_num_cols], errors='coerce').notnull()]

                # Remove rows with missing values
                df_gen = df_gen.drop(df_gen[df_gen.isna().any(axis=1)].index)
                
                # Update process bar
                pbar.update(df_gen.shape[0] - already_generated)
                already_generated = df_gen.shape[0]
                
        return df_gen.head(n_samples)
    
    
    def great_sample(self, starting_prompts: tp.Union[str, list[str]], temperature=0.7, device="cuda"):
        """ Generate samples conditioned on an arbitrary input.
            :param starting_prompts: String or List of Strings on which the output is conditioned. For example, "Sex is female, Age is 26".
            :param temperature: The generation samples each token from the probility distribution given by a softmax function. The temperature parameter controlls the softmax function. A low temperature makes it sharper (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output. See this blog articel (https://huggingface.co/blog/how-to-generate) to read more about the generation process.
            :param device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.
            
            ToDo: Set n_samples to generate more samples for one conditional input.
        """
        self.model.to(device)
        
        starting_prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        
        generated_data = []
        
        # Generate a sample for each starting point
        for prompt in starting_prompts:
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)
            
            # Generate tokens
            gen = self.model.generate(input_ids=torch.unsqueeze(start_token, 0), max_length=self.max_length+20,
                                      do_sample=True, temperature=temperature, pad_token_id=50256)
            
            generated_data.append(torch.squeeze(gen))
           
        # Convert Text back to Tabular Data
        decoded_data = convert_tokens_to_text(generated_data, self.tokenizer)
        print(decoded_data)
        df_gen = convert_text_to_tabular_data(decoded_data, pd.DataFrame(columns=self.columns))
                
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
            if isinstance(attributes["target_col_dist"], np.ndarray):
                attributes["target_col_dist"] = list(attributes["target_col_dist"])
            
            #print(attributes)
            
            json.dump(attributes, f)
            
        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")
        
    def load_finetuned_model(self, path: str):
        """ Load the weights of a fine-tuned large language model into the GReaT pipeline
            :param path: Path to the fine-tuned model
        """
        great.model.load_state_dict(torch.load(path))
        
    
    @classmethod
    def load_from_dir(cls, path: str):
        """ Load GReaT class from directory
            :param path: Directory where model is saved
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."
        
        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)
        
        # Create new GReaT model instance
        great = cls(attributes["llm"])
                    
        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)
            
        # Load model weights
        great.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))
        
        return great
        
     
    

def convert_tokens_to_text(tokens: list[list[int]], tokenizer: AutoTokenizer):
    
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]
    
    return text_data
    
    
    
def convert_text_to_tabular_data(text: list[str], df_gen: pd.DataFrame):
    
    columns = df_gen.columns.to_list()
    
    # Convert text to tabular data
    for t in text:
        features = t.split(",")

        td = dict.fromkeys(columns)

        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" ")

            if values[0] in columns and not td[values[0]]:
                td[values[0]] = [" ".join(values[2:])]

        df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
        
    return df_gen