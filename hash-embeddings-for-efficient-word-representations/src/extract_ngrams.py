"""
Standalone script provided to generate top 1M n-grams corresponding to 
each document. The train and test documents are converted to bag of these
n-grams. Successfully created ngrams are stored as pickle objects in 
data/ngrams
"""
import re
import operator
import string
import heapq
import pickle
from collections import defaultdict
import time
import random

import progressbar

#name, n = "ag_news", 10
#name, n = "amazon_review_full", 2
name, n = "amazon_review_polarity", 1
#name, n = "dbpedia", 10
#name, n = "yahoo_answers", 6
#name, n = "yelp_review_full", 6
#name, n = "yelp_review_polarity", 6

data = pickle.load(
    open('../data/preprocessed/{}_csv_train.pkl'.format(name), 'rb'))
ngrams = defaultdict(int)
punct_patt = re.compile('[%s]' % re.escape(string.punctuation))


def remove_punct(in_string):
    return re.sub(punct_patt, ' ', in_string)


bar = progressbar.ProgressBar()
data_cleaned = [None]*len(data)
print("train length: {}".format(len(data)))
#
#
def get_line(d):
    if 'text' in d:
        if 'title' in d:
            line = d['text'] + " " + d['title']
        else:
            line = d['text']
    else:
        line = d['question'] + " " + d['answer'] + " " + d['title']

    return line


for i, d in bar(enumerate(data)):
    line = get_line(d)
    words = remove_punct(line.lower()).split()
    data_cleaned[i] = (words, d['class'])
    for idx in range(len(words)):
        for ngram in range(1, n):
            if idx+ngram <= len(words):
                ngrams['_'.join(words[idx:idx+ngram])] += 1

print("Ngrams constructed")

nlargest = heapq.nlargest(1000000, ngrams.items(), key=operator.itemgetter(1))
nlargest_dict = dict()
for ng in nlargest:
    nlargest_dict[ng[0]] = len(nlargest_dict)+1
print("n-largest selected")

t1 = time.time()
del ngrams
del data
t2 = time.time()
print("Time to delete ngrams: {}".format(t2-t1))
max_len = 100
bar = progressbar.ProgressBar()
train_doc2id = [None]*len(data_cleaned)

for i, (d, c) in bar(enumerate(data_cleaned)):
    train_doc2id[i] = (set(), c)
    temp = set()
    for idx in range(len(d)):
        for ngram in range(1, n):
            if idx+ngram <= len(d):
                key = '_'.join(d[idx:idx+ngram])
                if key in nlargest_dict:
                    temp.add(nlargest_dict[key])
                else:
                    break
            else:
                break

    if len(temp) > max_len:
        train_doc2id[i][0].update(set(random.sample(temp, max_len)))
    else:
        train_doc2id[i][0].update(set(temp))

print("train docs vectorized")
pickle.dump(train_doc2id, open(
    '../data/ngrams/{}_csv_train.pkl'.format(name), 'wb'))

t1 = time.time()
del train_doc2id
del data_cleaned
t2 = time.time()
print("Time to delete train elements: {}".format(t2-t1))

data = pickle.load(
    open('../data/preprocessed/{}_csv_test.pkl'.format(name), 'rb'))
print("test length: {}".format(len(data)))
test_doc2id = [None]*len(data)
bar = progressbar.ProgressBar()
for i, d in bar(enumerate(data)):
    line = get_line(d)
    words = remove_punct(line.lower()).split()
    test_doc2id[i] = (set(), d['class'])
    temp = set()
    for idx in range(len(words)):
        for ngram in range(1, n):
            if idx+ngram <= len(words):
                key = '_'.join(words[idx:idx+ngram])
                if key in nlargest_dict:
                    temp.add(nlargest_dict[key])
                else:
                    break
            else:
                break
    if len(temp) > max_len:
        test_doc2id[i][0].update(set(random.sample(temp, max_len)))
    else:
        test_doc2id[i][0].update(set(temp))

print("test docs vectorized")
pickle.dump(test_doc2id, open(
    '../data/ngrams/{}_csv_test.pkl'.format(name), 'wb'))
