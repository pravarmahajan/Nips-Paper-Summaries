"""
This is the main script for replication of the results of the paper 
"Hash Embeddings for Efficient Word Representations", a NIPS 2017 paper.
The best way to run this script is via 'run.sh' provided in the parent
directory, otherwise path references need to be fixed.

usage: python main.py [-h]
               [-ds {agnews,dbpedia,yelp_pol,yelp_full,yahoo,amazon_full,amazon_pol}]
               [-K VOCAB_SIZE] [-k NUM_HASH] [-B BUCKET] [-d EMBEDDING]
               [-hi [HIDDEN [HIDDEN ...]]] [-lr LEARNING_RATE] [-e NUM_EPOCHS]
               [-b BATCH_SIZE] [-g] [-emb {std,hash}] [-dict]

optional arguments:
  -h, --help            show this help message and exit
  -ds {agnews,dbpedia,yelp_pol,yelp_full,yahoo,amazon_full,amazon_pol},
      --dataset {agnews,dbpedia,yelp_pol,yelp_full,yahoo,amazon_full,amazon_pol}
  -K VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        size of the vocabulary
  -k NUM_HASH, --num_hash NUM_HASH
                        number of hash functions
  -B BUCKET, --bucket BUCKET
                        number of embedding buckets
  -d EMBEDDING, --embedding EMBEDDING
                        embedding vector size
  -H [HIDDEN [HIDDEN ...]], --hidden [HIDDEN [HIDDEN ...]]
                        number of hidden units
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        number of epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        size of a minibatch
  -g, --use_gpu         use gpu?
  -emb {std,hash}, --embedding_type {std,hash}
  -dict, --with_dict
"""

import progressbar
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable

import args
import dataloader
import dataset
from models import HashEmbedding, StandardEmbedding, ClassifierModel


def main():
    """
    The main function
    - load the dataset
    - create the embedding model
    - train the classifier model on the dataset based on
      the type of embedding model
    """
    parsed_args = args.parse_arguments()
    agg_function = torch.sum
    use_cuda = parsed_args.use_gpu and torch.cuda.is_available()
    max_len = 100 # All the documents will be clipped/zero padded to this length
    val_frac = 0.05 # Validation fraction

    """Loading dataset"""
    if not parsed_args.with_dict:
        dl_obj = dataloader.UniversalArticleDatasetProvider(
            parsed_args.dataset, valid_fraction=val_frac)
        dl_obj.load_data()
        train_dataloader, val_dataloader, test_dataloader = dataset.create_dataset_nodict(
            dl_obj, parsed_args.vocab_size, parsed_args.batch_size, use_cuda, max_len)
        num_classes = dl_obj.num_classes
    else:
        train_dataloader, val_dataloader, test_dataloader, num_classes = dataset.create_dataset_wdict(
            parsed_args.dataset, val_frac, parsed_args.batch_size, use_cuda, max_len)

    """Creating the embedding model"""
    if parsed_args.embedding_type == "hash":
        embedding_model = HashEmbedding(parsed_args.vocab_size, parsed_args.num_hash,
                                        parsed_args.bucket, parsed_args.embedding, agg_function, use_cuda)
    else:
        embedding_model = StandardEmbedding(
            parsed_args.vocab_size, parsed_args.embedding, use_cuda)

    embedding_model = embedding_model.cuda() if use_cuda else embedding_model
    embedding_model.initializeWeights()

    model = ClassifierModel(embedding_model, num_classes,
                            parsed_args.hidden, use_cuda)
    model = model.cuda() if use_cuda else model
    model.initializeWeights()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda() if use_cuda else criterion

    prev_loss = float("inf")

    """Train the classification model"""
    backoff_attempts = 0
    for _ in range(parsed_args.num_epochs):
        bar = progressbar.ProgressBar()
        print("Epoch = {}".format(_))

        optimizer = torch.optim.Adam(list(model.parameters())+list(embedding_model.parameters()),
                                     lr=parsed_args.learning_rate)

        for (i, d) in bar(enumerate(train_dataloader)):
            t = Variable(d[1])
            t = t.cuda() if use_cuda else t
            d[0] = d[0].cuda() if use_cuda else d[0]
            output = model(d[0])
            loss = criterion(output, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del d
            del t
            del output

        correct = 0
        total = 0
        val_loss = 0
        for (i, d) in enumerate(val_dataloader):
            t = d[1].cuda() if use_cuda else d[1]
            d[0] = d[0].cuda() if use_cuda else d[0]
            output = model(d[0])
            pred = output.max(1)[1].data
            correct = correct + (pred == t).sum()
            total += pred.size(0)
            val_loss += criterion(output, Variable(t)).data[0]
            del d
            del output

        print("Val accuracy = {:.2f}".format(correct*100/total))
        print("Val loss = {:.2f}".format(val_loss))

        if prev_loss < val_loss: # If current loss is more than prev loss, then we stop further training
                             # and take the last state of the model as our trained model.
            model = prev_model
            print("Early Stopping")
            break
        else:
            prev_loss = val_loss
            prev_model = copy.deepcopy(model)

    correct = 0
    total = 0
    for (i, d) in enumerate(test_dataloader):
        d[0] = d[0].cuda() if use_cuda else d[0]
        t = d[1].cuda() if use_cuda else d[1]
        pred = model(d[0]).max(1)[1].data
        correct = correct + (pred == t).sum()
        total += pred.size(0)
        del d
        del t

    print("Test Accuracy = {:.2f}".format(correct*100/total))

main()
