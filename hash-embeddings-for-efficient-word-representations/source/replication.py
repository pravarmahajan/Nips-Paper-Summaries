
# coding: utf-8

# In[1]:


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
import dataloader
import random
import progressbar


# In[2]:


dl_obj = dataloader.UniversalArticleDatasetProvider(1, valid_fraction=0.05)
dl_obj.load_data()


# In[3]:


def remove_punct(in_string):
    return ''.join([ch.lower() if ch not in string.punctuation else ' ' for ch in in_string])


def bigram_vectorizer(documents, bigram_dict={}):
    if len(bigram_dict)==0:
        docs2id = [None]*len(documents)
        for (i, document) in enumerate(documents):
            tokens = document.split(' ')
            docs2id[i] = [None]*(len(tokens)-1)
            for j in range(len(tokens)-1):
                key = tokens[j]+"_"+tokens[j+1]
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict)+1
                docs2id[i][j] = bigram_dict[key]
        return bigram_dict, docs2id
    else:
        docs2id = [None]*len(documents)
        for (i, document) in enumerate(documents):
            tokens = document.split(' ')
            docs2id[i] = [None]*(len(tokens)-1)
            for j in range(len(tokens)-1):
                key = tokens[j]+"_"+tokens[j+1]
                if key not in bigram_dict:
                    docs2id[i][j] = 0
                else:
                    docs2id[i][j] = bigram_dict[key]
        return docs2id


# In[4]:


def input_dropout(docs_as_ids, min_len=4, max_len=100):
    dropped_input = [None]*len(docs_as_ids)
    for i, doc in enumerate(docs_as_ids):
        random_len = random.randrange(min_len, max_len+1)
        idx = max(len(doc)-random_len, 0)
        dropped_input[i] = doc[idx:idx+random_len]
    return dropped_input


# In[5]:


max_words = 10**7
num_hash = 2
num_buckets = 10**6
embedding_dim = 20
num_classes = 4
num_hidden_units = 50
learning_rate = 1e-3
agg_function = torch.sum
num_epochs = 10
batch_size = 1024
max_len = 150
use_cuda = torch.cuda.is_available()

# In[6]:


train_documents = [remove_punct(sample['text']) for sample in dl_obj.train_samples]
train_targets = [sample['class'] - 1 for sample in dl_obj.train_samples]

val_documents = [remove_punct(sample['text']) for sample in dl_obj.valid_samples]
val_targets = [sample['class'] - 1 for sample in dl_obj.valid_samples]

bigram_dict, train_docs2id = bigram_vectorizer(train_documents)
val_docs2id = bigram_vectorizer(val_documents, bigram_dict)

train_docs2id = input_dropout(train_docs2id)
train_docs2id = torch.LongTensor([d+[0]*(max_len-len(d)) for d in train_docs2id])
train_targets = torch.LongTensor(np.asarray(train_targets, 'int32'))

val_docs2id = input_dropout(val_docs2id)
val_docs2id = torch.LongTensor([d+[0]*(max_len-len(d)) if max_len > len(d) else d[:max_len] for d in val_docs2id])
val_targets = torch.LongTensor(np.asarray(val_targets, 'int32'))

train_docs2id = train_docs2id % max_words
val_docs2id = val_docs2id % max_words

train_docs2id = train_docs2id.cuda() if use_cuda else train_docs2id
#train_targets = train_targets.unsqueeze(-1)
train_targets = train_targets.cuda() if use_cuda else train_targets

val_docs2id = val_docs2id.cuda() if use_cuda else val_docs2id
#val_targets = val_targets.unsqueeze(-1)
val_targets = val_targets.cuda() if use_cuda else val_targets

train_dataset = TensorDataset(train_docs2id, train_targets)
val_dataset = TensorDataset(val_docs2id, val_targets)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# In[7]:


class HashEmbedding(nn.Module):
    
    def __init__(self, num_words, num_hash_functions, num_buckets, embedding_size, agg_function):
        super(HashEmbedding, self).__init__()
        self.num_words = num_words # K
        self.num_hash_functions = num_hash_functions # k
        self.num_buckets = num_buckets # B
        self.embedding_size = embedding_size # d
        self.W = nn.Parameter(torch.FloatTensor(num_buckets, embedding_size)) # B x d
        self.agg_func = agg_function
        self.hash_table = torch.LongTensor(np.random.randint(0, 2**30,
                                size=(num_words, num_hash_functions)))%num_buckets # K x k
        
        self.P = nn.Parameter(torch.FloatTensor(num_words, num_hash_functions)) # K x k
        #self.W = self.W.cuda() if use_cuda else self.W
        #self.P = self.P.cuda() if use_cuda else self.P
        self.hash_table = self.hash_table.cuda() if use_cuda else self.hash_table

    
    def forward(self, words_as_ids):
        embeddings = []
        pvals = []
        #import ipdb; ipdb.set_trace()
        for i in range(self.num_hash_functions):
            hashes = torch.take(self.hash_table[:, i], words_as_ids)
            embeddings.append(self.W[hashes, :]*self.P[words_as_ids, :][:, :, i].unsqueeze(-1))
            pvals.append(self.P[hashes, :][:, :, i].unsqueeze(-1))

        cat_embeddings = torch.stack(embeddings, -1)
        cat_embeddings = self.agg_func(cat_embeddings, -1)
        cat_pvals = torch.cat(pvals, -1)
        output = torch.cat([cat_embeddings, cat_pvals], -1)
        output = output.cuda() if use_cuda else output
        return output
        #return cat_embeddings
    
    def initializeWeights(self):
        nn.init.normal(self.W, 0, 0.1)
        nn.init.normal(self.P, 0, 0.0005)
        
class Model(nn.Module):
    
    def __init__(self, embedding_model, num_classes, num_hidden_units):
        super(Model, self).__init__()
        self.embedding_model = embedding_model
        self.num_classes = num_classes
        self.dense_layer = nn.Linear(self.embedding_model.embedding_size+self.embedding_model.num_hash_functions,
        #self.dense_layer = nn.Linear(self.embedding_model.embedding_size,
                                     num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, num_classes)

        self.dense_layer = self.dense_layer.cuda() if use_cuda else self.dense_layer
        self.output_layer = self.output_layer.cuda() if use_cuda else self.output_layer
    
    def forward(self, words_as_ids):
        mask = Variable(torch.unsqueeze(1-torch.eq(words_as_ids, 0).float(), -1))
        embedded = torch.sum(self.embedding_model(words_as_ids)*mask, 1)
        #import pdb; pdb.set_trace()
        dense_output = F.relu(self.dense_layer(embedded))
        final_output = self.output_layer(dense_output)
        final_output = final_output.cuda() if use_cuda else final_output
        return final_output
    
    def initializeWeights(self):
        nn.init.xavier_uniform(self.dense_layer.weight)
        nn.init.xavier_uniform(self.output_layer.weight)
        model.dense_layer.bias.data.zero_()
        model.output_layer.bias.data.zero_()


# In[8]:


embedding_model = HashEmbedding(max_words, num_hash, num_buckets, embedding_dim, agg_function)
embedding_model = embedding_model.cuda() if use_cuda else embedding_model
embedding_model.initializeWeights()


# In[9]:


model = Model(embedding_model, num_classes, num_hidden_units)
model = model.cuda() if use_cuda else model
model.initializeWeights()


# In[10]:


criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda() if use_cuda else criterion
optimizer = torch.optim.Adam(list(model.parameters())+list(embedding_model.parameters()),
                             lr=learning_rate)

for _ in range(num_epochs):
    optimizer = torch.optim.Adam(list(model.parameters())+list(embedding_model.parameters()),
                                 lr=learning_rate)
    learning_rate = learning_rate*0.5
    bar = progressbar.ProgressBar()
    print("Epoch = {}".format(_))
    for (i, d) in bar(enumerate(train_dataloader)):
        data_point = d[0].cuda() if use_cuda else d[0]
        t = Variable(d[1])
        t = t.cuda() if use_cuda else t
        output = model(data_point)
        loss = criterion(output, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = 0
    total = 0
    for (i, d) in enumerate(val_dataloader):
        data_point = d[0].cuda() if use_cuda else d[0]
        t = d[1].cuda() if use_cuda else d[1]
        pred = model(data_point).max(1)[1].data
        correct = correct + (pred==t).sum()
        total += pred.size(0)
    print("Accuracy = {:.2f}".format(correct*100/total))

