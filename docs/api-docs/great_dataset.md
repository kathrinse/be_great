<!-- markdownlint-disable -->

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `great_dataset`






---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_dataset.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GReaTDataset`
GReaT Dataset 

The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step. 



**Attributes:**
 
 - <b>`tokenizer`</b> (AutoTokenizer):  Tokenizer from HuggingFace 


---



<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_dataset.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GReaTDataset.set_tokenizer`

```python
set_tokenizer(tokenizer)
```

Set the Tokenizer 



**Args:**
 
 - <b>`tokenizer`</b>:  Tokenizer from HuggingFace 


---

<a href="https://github.com/kathrinse/be_great/tree/main/be_great\great_dataset.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GReaTDataCollator`
GReaT Data Collator 

Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
