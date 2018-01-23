import pickle
import string
import random
import re

import nltk
import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import args
from models import *
import dataloader

args = args.parse_arguments()

ds = {
      "agnews": 1,
      "dbpedia": 4,
      "yelp_pol": 8,
      "yelp_full": 7,
      "yahoo": 6,
      "amazon_full": 2,
      "amazon_review": 3
     }

dl_obj = dataloader.UniversalArticleDatasetProvider(ds[args.dataset], valid_fraction=0.05)
dl_obj.load_data()

punct_patt = re.compile('[%s]' %re.escape(string.punctuation))

def remove_punct(in_string):
    return re.sub(punct_patt, ' ', in_string)

def bigram_vectorizer(documents):
    docs2id = [None]*len(documents)
    for (i, document) in enumerate(documents):
        tokens = document.split(' ')
        docs2id[i] = list([word_encoder(tokens[j]+"_"+tokens[j+1], args.vocab_size) for j in range(len(tokens)-1)])
    return docs2id

def word_encoder(w, max_idx):
    v = hash(w) 
    #v = int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)
    return (v % (max_idx-1)) + 1

def input_dropout(docs_as_ids, min_len=4, max_len=100):
    dropped_input = [None]*len(docs_as_ids)
    for i, doc in enumerate(docs_as_ids):
        random_len = random.randrange(min_len, max_len+1)
        idx = max(len(doc)-random_len, 0)
        dropped_input[i] = doc[idx:idx+random_len]
    return dropped_input

agg_function = torch.sum
max_len = 150
use_cuda = args.use_gpu and torch.cuda.is_available()
num_classes = 0

def create_dataset():
    global num_classes
    train_documents = [remove_punct(sample['title'] + " " + sample['text']) for sample in dl_obj.train_samples]
    train_targets = [sample['class'] - 1 for sample in dl_obj.train_samples]
    num_classes = max(train_targets)+1

    val_documents = [remove_punct(sample['title'] + " " + sample['text']) for sample in dl_obj.valid_samples]
    val_targets = [sample['class'] - 1 for sample in dl_obj.valid_samples]

    test_documents = [remove_punct(sample['title'] + " " + sample['text']) for sample in dl_obj.test_samples]
    test_targets = [sample['class'] - 1 for sample in dl_obj.test_samples]

    print("Done with loading")


    train_docs2id = bigram_vectorizer(train_documents)
    val_docs2id = bigram_vectorizer(val_documents)
    test_docs2id = bigram_vectorizer(test_documents)

    print("Vectorized")

    train_dataloader = create_dataloader(train_docs2id, train_targets, True, True)
    val_dataloader = create_dataloader(val_docs2id, val_targets)
    test_dataloader = create_dataloader(test_docs2id, test_targets)

    return train_dataloader, val_dataloader, test_dataloader

def create_dataloader(docs2id, targets, dropout = False, shuffle=False):
    if dropout:
        docs2id = input_dropout(docs2id)

    docs2id = torch.LongTensor([d+[0]*(max_len-len(d)) if max_len > len(d) else d[:max_len] for d in docs2id])
    targets = torch.LongTensor(np.asarray(targets, 'int32'))

    docs2id = docs2id.cuda() if use_cuda else docs2id
    targets = targets.cuda() if use_cuda else targets

    dataset = TensorDataset(docs2id, targets)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    return dataloader
    

def create_dataset2():
    global num_classes
    train_data = pickle.load(open('./data/ngrams/ag_news_csv_train.pkl', 'rb'))
    test_data = pickle.load(open('./data/ngrams/ag_news_csv_test.pkl', 'rb'))
    random.shuffle(train_data)

    idx = int(0.95 * len(train_data))
    train_docs2id = list([list(train_data[i][0]) for i in range(idx)])
    train_targets = list([train_data[i][1] for i in range(idx)])
    num_classes = max(train_targets)+1

    val_docs2id = list([list(train_data[i][0]) for i in range(idx, len(train_data))])
    val_targets = list([train_data[i][1] for i in range(idx, len(train_data))])

    test_docs2id = list([list(t[0]) for t in test_data])
    test_targets = list([t[1] for t in test_data])
    
    train_dataloader = create_dataloader(train_docs2id, train_targets, True, False)
    val_dataloader = create_dataloader(val_docs2id, val_targets)
    test_dataloader = create_dataloader(test_docs2id, test_targets)

    return train_dataloader, val_dataloader, test_dataloader

train_dataloader, val_dataloader, test_dataloader = create_dataset()

if args.embedding_type == "hash":
    embedding_model = HashEmbedding(args.vocab_size, args.num_hash, args.bucket, args.embedding, agg_function, use_cuda)
else:
    embedding_model = StandardEmbedding(args.vocab_size, args.embedding, use_cuda)
embedding_model = embedding_model.cuda() if use_cuda else embedding_model
embedding_model.initializeWeights()

model = ClassifierModel(embedding_model, num_classes, args.hidden, use_cuda)
model = model.cuda() if use_cuda else model
model.initializeWeights()

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda() if use_cuda else criterion

prev_loss = float("inf")
for _ in range(args.num_epochs):
    bar = progressbar.ProgressBar()
    print("Epoch = {}".format(_))

    optimizer = torch.optim.Adam(list(model.parameters())+list(embedding_model.parameters()),
                                 lr=args.learning_rate)

    for (i, d) in bar(enumerate(train_dataloader)):
        if i==20:
            break
        t = Variable(d[1])
        t = t.cuda() if use_cuda else t
        output = model(d[0])
        loss = criterion(output, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = 0
    total = 0
    loss = 0
    for (i, d) in enumerate(val_dataloader):
        t = d[1].cuda() if use_cuda else d[1]
        output = model(d[0])
        pred = output.max(1)[1].data
        correct = correct + (pred==t).sum()
        total += pred.size(0)
        loss += criterion(output, Variable(t))

    loss = loss[0].data[0]
    print("Val accuracy = {:.2f}".format(correct*100/total))
    print("Val loss = {:.2f}".format(loss))
    if prev_loss < loss:
        print("Early Stopping")
        model = prev_model
    else:
        prev_loss = loss
        prev_model = model

for (i, d) in enumerate(test_dataloader):
    t = d[1].cuda() if use_cuda else d[1]
    pred = model(d[0]).max(1)[1].data
    correct = correct + (pred==t).sum()
    total += pred.size(0)
print("Accuracy = {:.2f}".format(correct*100/total))
