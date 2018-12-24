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

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))[0])

import pickle
from src.datasets import Datesets
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.preprocessing import sequence
from keras.models import Model, Sequential
import numpy as np
import keras.backend as K
from keras.utils import np_utils
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 每个GPU现存上界控制在50%以内
session = tf.Session(config=config)

KTF.set_session(session)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# config = tf.ConfigProto
# config.gpu_options.per_process_gpu_memory_fraction = 0.5


(raw_datas, labels) = Datesets.load_data("in_hospital")

raw_datas = [list(set(data.split(' '))) for data in raw_datas]

nb_class = max(labels) + 1
batch_size = 32
epochs = 50

train_size = int(len(labels) * 0.8)
train_x = raw_datas[:train_size]
train_y = labels[:train_size]
test_x = raw_datas[train_size:]
test_y = labels[train_size:]

w2id_path = '../data/word2id.pkl'

# without flatten
# max_length = max([len(doc.split(' ')) for usr in train_x for doc in usr])
# max_steps = max([len(_) for _ in train_x])   #　文书最多个数


max_length = max([len(usr) for usr in train_x])
# max_steps = max([len(_) for _ in train_x])   #　文书最多个数


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


def vecotr_list(sent_list, wordIndices):
    res = [word_indices[v] for v in sent_list if v in wordIndices]
    return res

def vector_sentence(sentence, wordIndices):
    sent_list = sentence.strip().split(' ')
    return vecotr_list(sent_list, wordIndices)

# train_x = [[vector_sentence(doc, word_indices) for doc in usr] for usr in train_x]
# test_x = [[vector_sentence(doc, word_indices) for doc in usr] for usr in test_x]
# train_x = [sequence.pad_sequences(_, max_length) for _ in train_x]
# test_x = [sequence.pad_sequences(_, max_length) for _ in test_x]

train_x = [vecotr_list(doc, word_indices) for doc in train_x]
test_x = [vecotr_list(doc, word_indices) for doc in test_x]
train_x = sequence.pad_sequences(train_x, max_length)
test_x = sequence.pad_sequences(test_x, max_length)

test_x = np.array(test_x)
test_y = np.array(test_y)
train_x = np.array(train_x)
train_y = np.array(train_y)
# train_y = np_utils.to_categorical(train_y, nb_class)
# test_y = np_utils.to_categorical(test_y, nb_class)

print(train_x[0].shape)

model = Sequential()
model.add(Embedding(len(word_indices), output_dim=64, name='embedding'))
model.add(Dropout(0.441))
model.add(Dense(100))
model.add(LSTM(279))
model.add(Dense(20))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='SGD',
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y))
model.save('in_hospital.h5')
