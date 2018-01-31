"""
Defines the following models:
- Standard Embedding
- Hash Embedding
- Classifier
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


class StandardEmbedding(nn.Module):
    """
    Standard Embedding Model.
    A simple wrapper around PyTorch's Embedding module.
    """

    def __init__(self, num_words, embedding_size, use_cuda):
        """
        Constructor for Embedding model.
        Arguments:
        num_words - number of words to be embedded (vocabulary size)
        embedding_size - dimension of the embedding vector
        use_cuda - true if the embedding layer should reside on cuda
        """
        super(StandardEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.num_hash_functions = 0
        self.embeddings = nn.Embedding(num_words, embedding_size)
        self.embeddings = self.embeddings.cuda() if use_cuda else self.embeddings

    def forward(self, words_as_ids):
        return self.embeddings(Variable(words_as_ids))

    def initializeWeights(self):
        nn.init.xavier_uniform(self.embeddings.weight)


class HashEmbedding(nn.Module):
    """
    Hash Embedding Model.
    """

    def __init__(self, num_words, num_hash_functions, num_buckets, embedding_size, agg_function, use_cuda):
        """
        Constructor.
        - Define a hash table which has `num_hash_functions` entries for each word.
        - Define an embedding matrix with embedding vector for each bucket
        - Define a weight matrix corresponding to the weights of each bucket on each
          hash function

        Arguments:
        num_words: Number of words (vocabularize size)
        num_hash_functions: Number of hash functions
        num_buckets: Number of embedding vector buckets
        embedding_size: Size of embedding vector
        agg_function: Function to aggregate different embedding vectors corresponding
          to each word. We are using torch.sum. Other options are concat, average, etc
        use_cuda: store the tensor in gpu or not.
        """
        super(HashEmbedding, self).__init__()
        self.num_words = num_words  # K
        self.num_hash_functions = num_hash_functions  # k
        self.num_buckets = num_buckets  # B
        self.embedding_size = embedding_size  # d
        self.W = nn.Parameter(torch.FloatTensor(
            num_buckets, embedding_size))  # B x d
        self.agg_func = agg_function
        self.hash_table = torch.LongTensor(np.random.randint(0, 2**30,
                                                             size=(num_words, num_hash_functions))) % num_buckets  # K x k

        self.P = nn.Parameter(torch.FloatTensor(
            num_words, num_hash_functions))  # K x k
        self.hash_table = self.hash_table.cuda() if use_cuda else self.hash_table

    def forward(self, words_as_ids):
        """
        Compute the embeddings corresponding to a document, given words as a bag
        of word ids.

        - Select appropriate hash functions from `self.hash_table`
        - Gather the embedding vectors corresponding to the hash functions
        - Weigh the embedding vectors using the `self.P` table
        - Aggregate the embeddings for each word
        - Concatenate the embeddings of all words together to produce
          a document embedding.
        """
        embeddings = []
        pvals = []

        for i in range(self.num_hash_functions):
            hashes = torch.take(self.hash_table[:, i], words_as_ids)
            embeddings.append(
                self.W[hashes, :]*self.P[words_as_ids, :][:, :, i].unsqueeze(-1))
            pvals.append(self.P[hashes, :][:, :, i].unsqueeze(-1))

        cat_embeddings = torch.stack(embeddings, -1)
        cat_embeddings = self.agg_func(cat_embeddings, -1)
        cat_pvals = torch.cat(pvals, -1)
        output = torch.cat([cat_embeddings, cat_pvals], -1)
        return output

    def initializeWeights(self):
        nn.init.normal(self.W, 0, 0.1)
        nn.init.normal(self.P, 0, 0.0005)


class ClassifierModel(nn.Module):
    """
    Classification model.
    Compute embeddings corresponding to a document, aggregate them and optionally
    pass them through one or more hidden layers. Finally, run a softmax classification
    on the embedded (or hidden) representation of the document.
    """

    def __init__(self, embedding_model, num_classes, num_hidden_units, use_cuda):
        """
        Arguments:
        embedding_model: An object of type HashEmbedding or StandardEmbedding
        num_classes: Number of classes, this will be the number of neurons in the softmax layer
        num_hidden_units: A list of numbers corresponding to the number of hidden units in each
          layer. [0] or an empty list corresponds to no hidden layers.
        """
        super(ClassifierModel, self).__init__()
        self.embedding_model = embedding_model
        self.num_classes = num_classes
        self.num_hidden_units = num_hidden_units
        if sum(num_hidden_units) > 0:
            layer0 = nn.Linear(self.embedding_model.embedding_size +
                               self.embedding_model.num_hash_functions, num_hidden_units[0])
            self.dense_layer = nn.ModuleList(
                [layer0] + [nn.Linear(num_hidden_units[i], num_hidden_units[i+1]) for i in range(len(num_hidden_units)-1)])
            self.output_layer = nn.Linear(num_hidden_units[-1], num_classes)
            self.dense_layer = self.dense_layer.cuda() if use_cuda else self.dense_layer
        else:
            self.output_layer = nn.Linear(
                self.embedding_model.embedding_size+self.embedding_model.num_hash_functions, num_classes)
            self.dense_layer = None

        self.output_layer = self.output_layer.cuda() if use_cuda else self.output_layer

    def forward(self, words_as_ids):
        """
        Apply the dense and output layers on document embeddings
        """
        mask = Variable(torch.unsqueeze(
            1-torch.eq(words_as_ids, 0).float(), -1))
        embedded = torch.sum(self.embedding_model(words_as_ids)*mask, 1)

        if self.dense_layer is not None:
            dense_output = embedded
            for layer in self.dense_layer:
                dense_output = F.relu(layer(dense_output))
            final_output = self.output_layer(dense_output)
        else:
            final_output = self.output_layer(embedded)

        return final_output

    def initializeWeights(self):
        if self.dense_layer is not None:
            for layer in self.dense_layer:
                nn.init.xavier_uniform(layer.weight)
                layer.bias.data.zero_()
        nn.init.xavier_uniform(self.output_layer.weight)
        self.output_layer.bias.data.zero_()
