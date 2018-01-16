
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
import re
# In[2]:


dl_obj = dataloader.UniversalArticleDatasetProvider(2, valid_fraction=0.05)
dl_obj.load_data()


# In[3]:

punct_patt = re.compile('[%s]' %re.escape(string.punctuation))

def remove_punct(in_string):
    return re.sub(punct_patt, ' ', in_string)

def bigram_vectorizer(documents):
    docs2id = [None]*len(documents)
    for (i, document) in enumerate(documents):
        tokens = document.split(' ')
        docs2id[i] = [None]*(len(tokens)-1)
        for j in range(len(tokens)-1):
            key = tokens[j]+"_"+tokens[j+1]
            idx = word_encoder(key, max_words)
            docs2id[i][j] = idx
    return docs2id


# In[4]:


def word_encoder(w, max_idx):
    # v = hash(w) #
    v = int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)
    return (v % (max_idx-1)) + 1

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
num_hidden_units = 0
learning_rate = 1e-3
agg_function = torch.sum
num_epochs = 10
batch_size = 1024
max_len = 150
use_cuda = torch.cuda.is_available()

# In[6]:

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

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        self.hash_table = self.hash_table.cuda() if use_cuda else self.hash_table

    
    def forward(self, words_as_ids):
        embeddings = []
        pvals = []

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
    
    def initializeWeights(self):
        nn.init.normal(self.W, 0, 0.1)
        nn.init.normal(self.P, 0, 0.0005)
        
class StandardEmbedding(nn.Module):
    
    def __init__(self, num_words, embedding_size):
        super(StandardEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.num_hash_functions = 0
        self.embeddings = nn.Embedding(num_words, embedding_size)
    
    def forward(self, words_as_ids):
        return self.embeddings(words_as_ids)
    
    def initializeWeights(self):
        nn.init.xavier_uniform(self.embeddings.weight)

class Model(nn.Module):
    
    def __init__(self, embedding_model, num_classes, num_hidden_units):
        super(Model, self).__init__()
        self.embedding_model = embedding_model
        self.num_classes = num_classes
        if num_hidden_units > 0:
            self.dense_layer = nn.Linear(self.embedding_model.embedding_size+self.embedding_model.num_hash_functions,
                                         num_hidden_units)
            self.output_layer = nn.Linear(num_hidden_units, num_classes)

            self.dense_layer = self.dense_layer.cuda() if use_cuda else self.dense_layer
        else:
            self.output_layer = nn.Linear(self.embedding_model.embedding_size+self.embedding_model.num_hash_functions,
                                          num_classes)

        self.output_layer = self.output_layer.cuda() if use_cuda else self.output_layer
    
    def forward(self, words_as_ids):
        mask = torch.unsqueeze(1-torch.eq(words_as_ids, 0).float(), -1)
        embedded = torch.sum(self.embedding_model(words_as_ids)*mask, 1)

        if num_hidden_units > 0:
            dense_output = F.relu(self.dense_layer(embedded))
            final_output = self.output_layer(dense_output)
        else:
            final_output = self.output_layer(embedded)

        final_output = final_output.cuda() if use_cuda else final_output
        return final_output
    
    def initializeWeights(self):
        if num_hidden_units > 0:
            nn.init.xavier_uniform(self.dense_layer.weight)
            model.dense_layer.bias.data.zero_()
        nn.init.xavier_uniform(self.output_layer.weight)
        model.output_layer.bias.data.zero_()


# In[8]:


#embedding_model = HashEmbedding(max_words, num_hash, num_buckets, embedding_dim, agg_function)
embedding_model = StandardEmbedding(max_words, embedding_dim)
embedding_model = embedding_model.cuda() if use_cuda else embedding_model
embedding_model.initializeWeights()


# In[9]:


model = Model(embedding_model, num_classes, num_hidden_units)
model = model.cuda() if use_cuda else model
model.initializeWeights()


# In[10]:


criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda() if use_cuda else criterion

for _ in range(num_epochs):
    bar = progressbar.ProgressBar()
    print("Epoch = {}".format(_))

    optimizer = torch.optim.Adam(list(model.parameters())+list(embedding_model.parameters()),
                                 lr=learning_rate)

    for (i, d) in bar(enumerate(train_dataloader)):
        t = Variable(d[1])
        t = t.cuda() if use_cuda else t
        output = model(Variable(d[0]))
        loss = criterion(output, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = 0
    total = 0
    for (i, d) in enumerate(val_dataloader):
        t = d[1].cuda() if use_cuda else d[1]
        pred = model(Variable(d[0])).max(1)[1].data
        correct = correct + (pred==t).sum()
        total += pred.size(0)
    print("Accuracy = {:.2f}".format(correct*100/total))

for (i, d) in enumerate(test_dataloader):
    t = d[1].cuda() if use_cuda else d[1]
    pred = model(Variable(d[0])).max(1)[1].data
    correct = correct + (pred==t).sum()
    total += pred.size(0)
print("Accuracy = {:.2f}".format(correct*100/total))
