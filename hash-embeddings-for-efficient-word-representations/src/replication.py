import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import nltk
import numpy as np
import hashlib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import string
import random
import progressbar
import re
import cProfile, io, pstats
import args

from models import *
import dataloader

args = args.parse_arguments()

dl_obj = dataloader.UniversalArticleDatasetProvider(1, valid_fraction=0.05)
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

train_docs2id = input_dropout(train_docs2id)
train_docs2id = torch.LongTensor([d+[0]*(max_len-len(d)) for d in train_docs2id])
train_targets = torch.LongTensor(np.asarray(train_targets, 'int32'))

val_docs2id = torch.LongTensor([d+[0]*(max_len-len(d)) if max_len > len(d) else d[:max_len] for d in val_docs2id])
val_targets = torch.LongTensor(np.asarray(val_targets, 'int32'))

test_docs2id = torch.LongTensor([d+[0]*(max_len-len(d)) if max_len > len(d) else d[:max_len] for d in test_docs2id])
test_targets = torch.LongTensor(np.asarray(test_targets, 'int32'))

train_docs2id = train_docs2id.cuda() if use_cuda else train_docs2id
train_targets = train_targets.cuda() if use_cuda else train_targets

val_docs2id = val_docs2id.cuda() if use_cuda else val_docs2id
val_targets = val_targets.cuda() if use_cuda else val_targets

test_docs2id = test_docs2id.cuda() if use_cuda else test_docs2id
test_targets = test_targets.cuda() if use_cuda else test_targets

train_dataset = TensorDataset(train_docs2id, train_targets)
val_dataset = TensorDataset(val_docs2id, val_targets)
test_dataset = TensorDataset(test_docs2id, test_targets)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



#embedding_model = HashEmbedding(args.vocab_size, args.num_hash, args.bucket, args.embedding, agg_function, use_cuda)
embedding_model = StandardEmbedding(args.vocab_size, args.embedding, use_cuda)
embedding_model = embedding_model.cuda() if use_cuda else embedding_model
embedding_model.initializeWeights()





model = ClassifierModel(embedding_model, num_classes, args.hidden, use_cuda)
model = model.cuda() if use_cuda else model
model.initializeWeights()





criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda() if use_cuda else criterion

for _ in range(args.num_epochs):
    bar = progressbar.ProgressBar()
    print("Epoch = {}".format(_))

    optimizer = torch.optim.Adam(list(model.parameters())+list(embedding_model.parameters()),
                                 lr=args.learning_rate)

    for (i, d) in bar(enumerate(train_dataloader)):
        t = Variable(d[1])
        t = t.cuda() if use_cuda else t
        output = model(d[0])
        loss = criterion(output, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = 0
    total = 0
    for (i, d) in enumerate(val_dataloader):
        t = d[1].cuda() if use_cuda else d[1]
        pred = model(d[0]).max(1)[1].data
        correct = correct + (pred==t).sum()
        total += pred.size(0)
    print("Accuracy = {:.2f}".format(correct*100/total))

for (i, d) in enumerate(test_dataloader):
    t = d[1].cuda() if use_cuda else d[1]
    pred = model(d[0]).max(1)[1].data
    correct = correct + (pred==t).sum()
    total += pred.size(0)
print("Accuracy = {:.2f}".format(correct*100/total))
