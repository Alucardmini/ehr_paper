#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@software: PyCharm Community Edition
@file: deep_learn_process.py
@time: 12/21/18 3:31 PM
"""
import os
import sys
import pickle
from src.datasets import Datesets
from keras.layers import Embedding, Dense, LSTM

(raw_datas, labels) = Datesets.load_data("in_hospital")

nb_class = max(labels)

train_size = int(len(labels) * 0.8)
train_x = raw_datas[:train_size]
train_y = labels[:train_size]
test_x = raw_datas[train_size:]
test_y = labels[train_size:]

w2id_path = '../data/word2id.pkl'

max_length = max([len(doc.split(' ')) for usr in train_x for doc in usr])
max_steps = max([len(_) for _ in train_x])   #　文书最多个数

if not os.path.exists(w2id_path):
    words = set()
    for docs in train_x:
        for record in docs:
            record_list = record.split(' ')
            words = (words | set(record_list))
    word_indices = {v: i for i, v in enumerate(words)}
    pickle.dump(word_indices, open(w2id_path, 'wb+'))
else:
    word_indices = pickle.load(open(w2id_path, 'rb'))


def vector_sentence(sentence, wordIndices):
    sent_list = sentence.strip().split(' ')
    res = [word_indices[v] for v in sent_list if v in wordIndices]
    return res

train_x = [[vector_sentence(doc, word_indices) for doc in usr] for usr in train_x]
test_x = [[vector_sentence(doc, word_indices) for doc in usr] for usr in test_x]




