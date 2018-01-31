"""
Preprocess documents and create dataloader objects which can be used while training
Following preprocessing is performed:
- Remove all punctuations, as defined by string.punctuation, and substitute
    them by space.
- input dropout: randomly select consecutive bigrams between size 4 and 100.
- to compute word ids, we simply compute the hash() of each word
"""
import os
import re
import string
import pickle
import random

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

"""regex defined to identify punctuations"""
punct_pattern = re.compile('[%s]' % re.escape(string.punctuation))


def remove_punct(in_string):
    """Use regex to remove punctuations and substitute
    them with spaces."""
    return re.sub(punct_pattern, ' ', in_string.lower())


def bigram_vectorizer(documents, vocab_size):
    """Generates bigrams for each document and then encodes
    the using the word_encoder function"""
    docs2id = [None]*len(documents)
    for (i, document) in enumerate(documents):
        tokens = document.split(' ')
        docs2id[i] = list([word_encoder(tokens[j]+"_"+tokens[j+1], vocab_size)
                           for j in range(len(tokens)-1)])
    return docs2id


def word_encoder(w, vocab_size):
    """Computes a hash value for each word and bounds them
    between 1 and vocab_size"""
    v = hash(w)
    return (v % (vocab_size-1)) + 1


def input_dropout(docs_as_ids, min_len=4, max_len=100):
    """Performs input dropout, as described in section 5.1 of the paper"""
    dropped_input = [None]*len(docs_as_ids)
    for i, doc in enumerate(docs_as_ids):
        random_len = random.randrange(min_len, max_len+1)
        idx = max(len(doc)-random_len, 0)
        dropped_input[i] = doc[idx:idx+random_len]
    return dropped_input


def get_line(d):
    if 'text' in d:
        if 'title' in d:
            line = d['text'] + " " + d['title']
        else:
            line = d['text']
    else:
        line = d['question'] + " " + d['answer'] + " " + d['title']

    return line


def create_dataset_nodict(dl_obj, vocab_size, batch_size, use_cuda, max_len):
    """Create dataset without using dictionary (section 5.3)"""
    train_documents = [remove_punct(get_line(sample))
                       for sample in dl_obj.train_samples]
    train_targets = [sample['class'] - 1 for sample in dl_obj.train_samples]

    val_documents = [remove_punct(get_line(sample))
                     for sample in dl_obj.valid_samples]
    val_targets = [sample['class'] - 1 for sample in dl_obj.valid_samples]

    test_documents = [remove_punct(get_line(sample))
                      for sample in dl_obj.test_samples]
    test_targets = [sample['class'] - 1 for sample in dl_obj.test_samples]

    print("Done with loading")

    train_docs2id = bigram_vectorizer(train_documents, vocab_size)
    val_docs2id = bigram_vectorizer(val_documents, vocab_size)
    test_docs2id = bigram_vectorizer(test_documents, vocab_size)

    print("Vectorized")

    train_dataloader = create_dataloader(
        train_docs2id, train_targets, max_len, batch_size, use_cuda, True, True)
    val_dataloader = create_dataloader(
        val_docs2id, val_targets, max_len, batch_size, use_cuda)
    test_dataloader = create_dataloader(
        test_docs2id, test_targets, max_len, batch_size, use_cuda)

    return train_dataloader, val_dataloader, test_dataloader


def create_dataset_wdict(dataset, val_frac, batch_size, use_cuda, max_len):
    """Create dataset using a precomputed dictionary of n-grams (Section 5.4)"""
    if dataset == "agnews":
        pickle_train = 'ag_news_csv_train.pkl'
        pickle_test = 'ag_news_csv_test.pkl'
    elif dataset == "amazon_full":
        pickle_train = 'amazon_review_full_csv_train.pkl'
        pickle_test = 'amazon_review_full_csv_test.pkl'
    elif dataset == "amazon_pol":
        pickle_train = 'amazon_review_polarity_csv_train.pkl'
        pickle_test = 'amazon_review_polarity_csv_test.pkl'
    elif dataset == "dbpedia":
        pickle_train = 'dbpedia_csv_train.pkl'
        pickle_test = 'dbpedia_csv_test.pkl'
    elif dataset == "yahoo":
        pickle_train = 'yahoo_answers_csv_train.pkl'
        pickle_test = 'yahoo_answers_csv_test.pkl'
    elif dataset == "yelp_full":
        pickle_train = 'yelp_review_full_csv_train.pkl'
        pickle_test = 'yelp_review_full_csv_test.pkl'
    elif dataset == "yelp_pol":
        pickle_train = 'yelp_review_polarity_csv_train.pkl'
        pickle_test = 'yelp_review_polarity_csv_test.pkl'
    else:
        print("Incorrect dataset ".format(dataset))

    pickle_train = os.path.join("./data/ngrams/", pickle_train)
    train_data = pickle.load(open(pickle_train, 'rb'))
    random.shuffle(train_data)
    print("Train Data Loaded")

    idx = int((1-val_frac) * len(train_data))
    train_docs2id = list([list(train_data[i][0]) for i in range(idx)])
    train_targets = list([train_data[i][1] for i in range(idx)])

    num_classes = max(train_targets)+1

    val_docs2id = list([list(train_data[i][0])
                        for i in range(idx, len(train_data))])
    val_targets = list([train_data[i][1] for i in range(idx, len(train_data))])
    del train_data
    print("Train Data Processed")

    pickle_test = os.path.join("./data/ngrams/", pickle_test)
    test_data = pickle.load(open(pickle_test, 'rb'))
    print("Test Data Loaded")
    test_docs2id = list([list(t[0]) for t in test_data])
    test_targets = list([t[1] for t in test_data])
    del test_data
    print("Test Data Processed")

    train_dataloader = create_dataloader(
        train_docs2id, train_targets, max_len, batch_size, use_cuda, True, False)
    val_dataloader = create_dataloader(
        val_docs2id, val_targets, max_len, batch_size, use_cuda)
    test_dataloader = create_dataloader(
        test_docs2id, test_targets, max_len, batch_size, use_cuda)

    return train_dataloader, val_dataloader, test_dataloader, num_classes


def create_dataloader(docs2id, targets, max_len, batch_size, use_cuda=False, dropout=False, shuffle=False):
    """A wrapper function to generate dataloader objects"""
    if dropout:
        docs2id = input_dropout(docs2id)

    docs2id = torch.LongTensor(
        [d+[0]*(max_len-len(d)) if max_len > len(d) else d[:max_len] for d in docs2id])
    targets = torch.LongTensor(np.asarray(targets, 'int32'))

    #docs2id = docs2id.cuda() if use_cuda else docs2id
    #targets = targets.cuda() if use_cuda else targets

    dataset = TensorDataset(docs2id, targets)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
