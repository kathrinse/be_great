
[//]: # (![Screenshot]&#40;./imgs/GReaT_logo.png&#41;)
<p align="center">
<img src="./imgs/GReaT_logo.png" width="326"/>
</p>

<p align="center">
<b>Generation of Realistic Tabular Data</b>
</p>

### **tl&dr**

Our GReaT framework utilizes the capabilities of pretrained large language Transformer models to synthesize realistic tabular data. 
It follows the common data science API, and require only four lines of code to generate new samples. 


## GReaT Installation

The GReaT framework can be easily installed using with [pip](https://pypi.org/project/pip/): 
```bash
pip install be-great
```



## GReaT Quickstart
Here is an example 
```python
from be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)['data']

model = GReaT(llm='distilgpt2', epochs=50)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb)

## GReaT Citation 
If you use GReaT, please link or cite our work:
```tex
@article{
}
```

## GReaT Acknowledgements

We sincerely thank [HuggingFace](https://huggingface.co/) :hugs: framework. 
