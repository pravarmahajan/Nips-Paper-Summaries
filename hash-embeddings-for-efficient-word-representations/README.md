# Paper Replication - Hash Embeddings for Efficient Word Representations

This is a replication of a NIPS 2017 paper - [Hash Embeddings for Efficient Word Representations](https://papers.nips.cc/paper/7078-hash-embeddings-for-efficient-word-representations).

## Getting Started
You can clone the repository on your machine by issuing the following command on the terminal:
```
git clone https://github.com/pravarmahajan/Nips-Paper-Summaries.git
```

## Prerequisites

### Python Packages
The neural network architecture is built using the [PyTorch](http://pytorch.org/) framework, please refer to the installation instructions in that page to install PyTorch according to your system configuration.
Once PyTorch is up and running on your machine, for installing other dependencies via pip:
```
pip install -r requirements.txt
```

### Dataset
To download the datasets needed for running the experiments on, we have provided two scripts - `download_data.py` and `extract_ngrams.py`.
In the `data` directory, create two additional directories named `ngrams` and `preprocessed`.
```
cd data
mkdir ngrams
mkdir preprocessed
cd ..
```
Now execute the two scripts one after the other:
```
python download_data.py
python extract_ngrams.py
```
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

