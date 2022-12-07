
[//]: # (![Screenshot]&#40;https://github.com/kathrinse/be_great/blob/main/imgs/GReaT_logo.png&#41;)
<p align="center">
<img src="https://github.com/kathrinse/be_great/raw/main/imgs/GReaT_logo.png" width="326"/>
</p>

<p align="center">
<strong>Generation of Realistic Tabular data</strong>
<br> with pretrained Transformer-based language models
</p>

&nbsp;
&nbsp;
&nbsp;

Our GReaT framework utilizes the capabilities of pretrained large language Transformer models to synthesize realistic tabular data. 
New samples are generated with just a few lines of code, following an easy-to-use API. Please see our [publication](https://arxiv.org/abs/2210.06280) for more details. 

## GReaT Installation

The GReaT framework can be easily installed using with [pip](https://pypi.org/project/pip/) - requires a Python version >= 3.9: 
```bash
pip install be-great
```



## GReaT Quickstart

In the example below, we show how the GReaT approach is used to generate synthetic tabular data for the California Housing dataset.
```python
from be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

model = GReaT(llm='distilgpt2', batch_size=32, epochs=50)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kathrinse/be_great/blob/main/examples/GReaT_colab_example.ipynb)

## GReaT Citation 

If you use GReaT, please link or cite our work:

``` bibtex
@article{borisov2022language,
  title={Language Models are Realistic Tabular Data Generators},
  author={Borisov, Vadim and Se{\ss}ler, Kathrin and Leemann, Tobias and Pawelczyk, Martin and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2210.06280},
  year={2022}
}
```

## GReaT Acknowledgements

We sincerely thank the [HuggingFace](https://huggingface.co/) :hugs: framework. 
