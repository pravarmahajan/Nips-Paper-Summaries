# Paper Replication - Hash Embeddings for Efficient Word Representations

This is a replication of a NIPS 2017 paper - [Hash Embeddings for Efficient Word Representations](https://papers.nips.cc/paper/7078-hash-embeddings-for-efficient-word-representations). Please refer to the [replication report](replication_report.pdf) for the summary of our replication results.

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
python src/download_data.py
python src/extract_ngrams.py
```
Whereas `download.py` downloads all the datasets, `extract_ngrams.py` only works on the specified dataset in the same script. Change the dataset name to extract ngrams from each of those datasets.

## Running the Experiments

The easiest way to run the experiments is via the script `run.sh`. In the parent directory, type:
```
sh run.sh
```
Alternatively, you may execute the main script as:
```
python src/main.py [args]
```
Details on the optional arguments are available by executing the main script with `-h` flag:
```
python src/main.py -h
```
A few important arguments to the script:
* `-h`: Displays help
* `-ds`: Dataset to work on. Should be one of `agnews`, `dbpedia`, `yelp_pol`, `yelp_full`, `yahoo`, `amazon_full`, `amazon_pol`
* `-K`: Vocabulary size when using dictionary, size of the first hash layer in case of no dictionary
* `-k`: Number of hash functions
* `-B`: Number of embedding buckets
* `-d`: Embedding dimension. All words will be mapped into continuous vectors of this size.
* `-H`: Hidden layers, if any
* `-emb`: Hashing type, one of `std` and `hash`. Hash embedding uses the embedding model described in this paper, standard embedding generates embedding for each word individually.
* `-dict`: Toggle between dictionary based model or no dictionary model

## Contributing
Contributions to improve and enhance the codebase are always welcome. Feel free to submit a pull request :-)
## Authors
* [Pravar D Mahajan](https://pravarmahajan.github.io)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgements
* Many thanks to authors of [the paper ](https://papers.nips.cc/paper/7078-hash-embeddings-for-efficient-word-representations) - Dan Svenstrup, Jonas Meinertz Hansen and Ole Winther (all from the Technical University of Denmark)
* Also thanks to Dan Svenstrup for releasing the Keras based implementation of the work [here](https://github.com/dsv77/hashembedding/tree/master/HashEmbedding).

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

