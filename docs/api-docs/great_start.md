<!-- markdownlint-disable -->

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `great_start`






---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GReaTStart`
Abstract super class GReaT Start 

GReaT Start creates tokens to start the generation process. 



**Attributes:**
 
 - <b>`tokenizer`</b> (AutoTokenizer):  Tokenizer, automatically downloaded from llm-checkpoint 

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaTStart.__init__`

```python
__init__(tokenizer)
```

Initializes the super class. 



**Args:**
 
 - <b>`tokenizer`</b>:  Tokenizer from the HuggingFace library 




---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaTStart.get_start_tokens`

```python
get_start_tokens(n_samples: int) → List[List[int]]
```

Get Start Tokens 

Creates starting points for the generation process 



**Args:**
 
 - <b>`n_samples`</b>:  Number of start prompts to create 



**Returns:**
 List of n_sample lists with tokens 


---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CategoricalStart`
Categorical Starting Feature 

A categorical column with its categories is used as starting point. 



**Attributes:**
 
 - <b>`start_col`</b> (str):  Name of the categorical column 
 - <b>`population`</b> (list[str]):  Possible values the column can take 
 - <b>`weights`</b> (list[float]):  Probabilities for the individual categories 

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CategoricalStart.__init__`

```python
__init__(tokenizer, start_col: str, start_col_dist: dict)
```

Initializes the Categorical Start 



**Args:**
 
 - <b>`tokenizer`</b>:  Tokenizer from the HuggingFace library 
 - <b>`start_col`</b>:  Name of the categorical column 
 - <b>`start_col_dist`</b>:  Distribution of the categorical column (dict of form {"Cat A": 0.8, "Cat B": 0.2}) 




---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CategoricalStart.get_start_tokens`

```python
get_start_tokens(n_samples)
```






---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ContinuousStart`
Continuous Starting Feature 

A continuous column with some noise is used as starting point. 



**Attributes:**
 
 - <b>`start_col`</b> (str):  Name of the continuous column 
 - <b>`start_col_dist`</b> (list[float]):  The continuous column from the train data set 
 - <b>`noise`</b> (float):  Size of noise that is added to each value 
 - <b>`decimal_places`</b> (int):  Number of decimal places the continuous values have 

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ContinuousStart.__init__`

```python
__init__(
    tokenizer,
    start_col: str,
    start_col_dist: List[float],
    noise: float = 0.01,
    decimal_places: int = 5
)
```

Initializes the Continuous Start 



**Args:**
 
 - <b>`tokenizer`</b>:  Tokenizer from the HuggingFace library 
 - <b>`start_col`</b>:  Name of the continuous column 
 - <b>`start_col_dist`</b>:  The continuous column from the train data set 
 - <b>`noise`</b>:  Size of noise that is added to each value 
 - <b>`decimal_places`</b>:  Number of decimal places the continuous values have 




---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ContinuousStart.get_start_tokens`

```python
get_start_tokens(n_samples)
```






---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RandomStart`
Random Starting Features 

Random column names are used as start point. Can be used if no distribution of any column is known. 



**Attributes:**
 
 - <b>`all_columns`</b> (List[str]):  Names of all columns 

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RandomStart.__init__`

```python
__init__(tokenizer, all_columns: List[str])
```

Initializes the Random Start 



**Args:**
 
 - <b>`tokenizer`</b>:  Tokenizer from the HuggingFace library 
 - <b>`all_columns`</b>:  Names of all columns 




---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_start.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RandomStart.get_start_tokens`

```python
get_start_tokens(n_samples)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
