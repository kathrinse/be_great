<!-- markdownlint-disable -->

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `great`






---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GReaT`
GReaT Class 

The GReaT class handles the whole generation flow. It is used to fine-tune a large language model for tabular data, and to sample synthetic tabular data. 



**Attributes:**
 
 - <b>`llm`</b> (str):  [HuggingFace checkpoint](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) of a pretrained large language model, used a basis of our model 
 - <b>`tokenizer`</b> (AutoTokenizer):  Tokenizer, automatically downloaded from llm-checkpoint 
 - <b>`model`</b> (AutoModelForCausalLM):  Large language model, automatically downloaded from llm-checkpoint 
 - <b>`experiment_dir`</b> (str):  Directory, where the training checkpoints will be saved 
 - <b>`epochs`</b> (int):  Number of epochs to fine-tune the model 
 - <b>`batch_size`</b> (int):  Batch size used for fine-tuning 
 - <b>`train_hyperparameters`</b> (dict):  Additional hyperparameters added to the TrainingArguments used by the  HuggingFaceLibrary, see here the [full list](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) of all possible values
 - <b>`columns`</b> (list):  List of all features/columns of the tabular dataset 
 - <b>`num_cols`</b> (list):  List of all numerical features/columns of the tabular dataset 
 - <b>`conditional_col`</b> (str):  Name of a feature/column on which the sampling can be conditioned 
 - <b>`conditional_col_dist`</b> (dict | list):  Distribution of the feature/column specified by condtional_col 

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaT.__init__`

```python
__init__(
    llm: str,
    experiment_dir: str = 'trainer_great',
    epochs: int = 100,
    batch_size: int = 8,
    **train_kwargs
)
```

Initializes GReaT. 



**Args:**
 
 - <b>`llm`</b>:  [HuggingFace checkpoint](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) of a pretrained large language model, used as basis for our model 
 - <b>`experiment_dir`</b>:   Directory, where the training checkpoints will be saved 
 - <b>`epochs`</b>:  Number of epochs to fine-tune the model 
 - <b>`batch_size`</b>:  Batch size used for fine-tuning 
 - <b>`train_kwargs`</b>:  Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,  see here the [full list](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) of all possible values  




---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaT.fit`

```python
fit(
    data: Union[pandas.core.frame.DataFrame, numpy.ndarray],
    column_names: Optional[List[str]] = None,
    conditional_col: Optional[str] = None,
    resume_from_checkpoint: Union[bool, str] = False
) → GReaTTrainer
```

Fine-tune GReaT using tabular data. 



**Args:**
 
 - <b>`data`</b>:  Pandas DataFrame or Numpy Array that contains the tabular data 
 - <b>`column_names`</b>:  If data is Numpy Array, the feature names have to be defined. If data is Pandas DataFrame, the value is ignored 
 - <b>`conditional_col`</b>:  If given, the distribution of this column is saved and used as a starting point for the generation process later. If None, the last column is considered as conditional feature 
 - <b>`resume_from_checkpoint`</b>:  If True, resumes training from the latest checkpoint in the experiment_dir. If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!) 



**Returns:**
 GReaTTrainer used for the fine-tuning process 

---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaT.great_sample`

```python
great_sample(
    starting_prompts: Union[str, list[str]],
    temperature: float = 0.7,
    max_length: int = 100,
    device: str = 'cuda'
) → DataFrame
```

Generate synthetic tabular data samples conditioned on a given input. 



**Args:**
 
 - <b>`starting_prompts`</b>:  String or List of Strings on which the output is conditioned.  For example, "Sex is female, Age is 26" 
 - <b>`temperature`</b>:  The generation samples each token from the probability distribution given by a softmax  function. The temperature parameter controls the softmax function. A low temperature makes it sharper  (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output. (See this [blog article](https:/huggingface.co/blog/how-to-generate) to read more about the generation process.) 
 - <b>`max_length`</b>:  Maximal number of tokens to generate - has to be long enough to not cut any information 
 - <b>`device`</b>:  Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU. 



**Returns:**
 Pandas DataFrame with synthetic data generated based on starting_prompts 

---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaT.load_finetuned_model`

```python
load_finetuned_model(path: str)
```

Load fine-tuned model 

Load the weights of a fine-tuned large language model into the GReaT pipeline 



**Args:**
 
 - <b>`path`</b>:  Path to the fine-tuned model 

---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `GReaT.load_from_dir`

```python
load_from_dir(path: str)
```

Load GReaT class 

Load trained GReaT model from directory. 



**Args:**
 
 - <b>`path`</b>:  Directory where GReaT model is saved 



**Returns:**
 New instance of GReaT loaded from directory 

---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaT.sample`

```python
sample(
    n_samples: int,
    start_col: Optional[str] = '',
    start_col_dist: Optional[dict, list] = None,
    temperature: float = 0.7,
    k: int = 100,
    max_length: int = 100,
    device: str = 'cuda'
) → DataFrame
```

Generate synthetic tabular data samples 



**Args:**
 
 - <b>`n_samples`</b>:  Number of synthetic samples to generate 
 - <b>`start_col`</b>:  Feature to use as starting point for the generation process. If not given, the target  learned during the fitting is used as starting point 
 - <b>`start_col_dist`</b>:  Feature distribution of the starting feature. Should have the format "{F1:  p1, F2: p2, ...}" for discrete columns or be a list of possible values for continuous columns. If not given, the target distribution learned during the fitting is used as starting point 
 - <b>`temperature`</b>:  The generation samples each token from the probability distribution given by a softmax  function. The temperature parameter controls the softmax function. A low temperature makes it sharper  (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output. (See this [blog article](https:/huggingface.co/blog/how-to-generate) to read more about the generation process.) 
 - <b>`k`</b>:  Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly 
 - <b>`max_length`</b>:  Maximal number of tokens to generate - has to be long enough to not cut any information! 
 - <b>`device`</b>:  Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU 



**Returns:**
 Pandas DataFrame with n_samples rows of generated data 

---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaT.save`

```python
save(path: str)
```

Save GReaT Model 

Saves the model weights and a configuration file in the given directory. 



**Args:**
 
 - <b>`path`</b>:  Path where to save the model 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
