import re
import operator
import string
import heapq
import pickle
from collections import defaultdict
import time

import progressbar

name = "ag_news"
n = 10
data = pickle.load(open('../data/preprocessed/{}_csv_train.pkl'.format(name), 'rb'))
ngrams = defaultdict(int)
punct_patt = re.compile('[%s]' %re.escape(string.punctuation))

def remove_punct(in_string):
    return re.sub(punct_patt, ' ', in_string)

bar = progressbar.ProgressBar()
data_cleaned = [None]*len(data)
print("train length: {}".format(len(data)))

def get_line(d):
    line = d['text'] + " " + d['title']
    #line = d['text']
    #line = d['question'] + " " + d['answer'] + " " + d['title']
    return line

for i, d in bar(enumerate(data)):
    line = get_line(d)
    words = remove_punct(line.lower()).split()
    data_cleaned[i] = (words, d['class'])
    for idx in range(len(words)):
        for ngram in range(1, n):
            if idx+ngram<=len(words):
                ngrams['_'.join(words[idx:idx+ngram])] += 1

print("Ngrams constructed")

nlargest = heapq.nlargest(1000000, ngrams.items(), key = operator.itemgetter(1))
nlargest_dict = dict()
for ng in nlargest:
    nlargest_dict[ng[0]] = len(nlargest_dict)+1
print("n-largest selected")

t1 = time.time()
del ngrams
t2 = time.time()
print("Time to delete ngrams: {}".format(t2-t1))

bar = progressbar.ProgressBar()
train_doc2id = [None]*len(data_cleaned)
for i, (d, c) in bar(enumerate(data_cleaned)):
    train_doc2id[i] = (set(), c)
    for idx in range(len(d)):
        for ngram in range(1, n):
            if idx+ngram <= len(d):
                key = '_'.join(d[idx:idx+ngram])
                if key in nlargest_dict:
                    train_doc2id[i][0].add(nlargest_dict[key])
                else:
                    break
            else:
                break

print("train docs vectorized")
pickle.dump(train_doc2id, open('../data/ngrams/{}_csv_train.pkl'.format(name), 'wb'))

t1 = time.time()
del train_doc2id
del data_cleaned
t2 = time.time()
print("Time to delete train elements: {}".format(t2-t1))

data = pickle.load(open('../data/preprocessed/{}_csv_test.pkl'.format(name), 'rb'))
print("test length: {}".format(len(data)))
test_doc2id = [None]*len(data)
bar = progressbar.ProgressBar()

for i, d in bar(enumerate(data)):
    line = get_line(d)
    words = remove_punct(line.lower()).split()
    test_doc2id[i] = (set(), d['class'])
    for idx in range(len(words)):
        for ngram in range(1, n):
            if idx+ngram <= len(words):
                key = '_'.join(words[idx:idx+ngram])
                if key in nlargest_dict:
                    test_doc2id[i][0].add(nlargest_dict[key])
                else:
                    break
            else:
                break

print("test docs vectorized")
pickle.dump(test_doc2id, open('../data/ngrams/{}_csv_test.pkl'.format(name), 'wb'))
