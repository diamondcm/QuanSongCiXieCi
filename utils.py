#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import logging
import random

import numpy as np
import json
from flags import parse_args


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)
    
def sentence_to_int_array(sentences, dictionary):
    
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index


def gen_dictionary(text):
    dic = {'n':0}
    max_index = 0
    for t in text:
        if t in dic:
            v = dic[t]
        else:
            max_index = max_index + 1
            v = max_index
            dic[t] = v
    
    reversed_dic = dict(zip(dic.values(), dic.keys()))
    return dic,reversed_dic, max_index
    
    
def get_train_data(vocabulary, batch_size, seq_length):
    # num_words_for_tgraining = 100000
    # text = vocabulary[:num_words_for_tgraining]
    
    # print(len(text))
    
    ##################
    # Your Code here
    ##################
    
    
    
    #dic,reversed_dic,max_index = gen_dictionary(vocabulary)
    
    data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, 10000)
    
    data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, len(count))
    logging.debug('vol len:' + str(len(count)))
    FLAGS, unparsed = parse_args()
    #f = open(FLAGS.dictionary, "wb")
    #j =json.dumps(dictionary)
    #print(j)
    #f.close()
    
    
    int_text = sentence_to_int_array(vocabulary,dictionary)
    #int_text = index_data(data,dictionary)
    
    # 计算有多少个batch可以创建
    n_batches = (len(int_text) // (batch_size * seq_length))
    
    # 计算每一步的原始数据，和位移一位之后的数据
    batch_origin = np.array(int_text[: n_batches * batch_size * seq_length])
    batch_shifted = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    # 将位移之后的数据的最后一位，设置成原始数据的第一位，相当于在做循环
    batch_shifted[-1] = batch_origin[0]
    
    batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, 1)
    batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, 1)

    batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))


    return batches,len(count)
    
    



def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
