#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

import numpy as np
import tensorflow as tf

import utils
from utils import read_data
from utils import build_dataset
from model import Model


from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


# with open(FLAGS.dictionary, encoding='utf-8') as inf:
    # dictionary = json.load(inf, encoding='utf-8')

# with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    # reverse_dictionary = json.load(inf, encoding='utf-8')

vocabulary = read_data(FLAGS.text)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, 10000)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, len(count))
     

#print(len(reverse_dictionary))    
# reverse_list = [reverse_dictionary[str(i)]
                # for i in range(len(reverse_dictionary))]
reverse_list = [reverse_dictionary[i]
                for i in range(len(reverse_dictionary))]
#titles = ['酒泉子','黄莺儿', '江神子', '蝶恋花', '渔家傲']
titles = ['华明明','木兰花', '夜半乐', '西平乐']

# logging.debug(len(count))
model = Model(learning_rate=FLAGS.learning_rate, batch_size=1, num_steps=3,num_words = len(count),rnn_layers = FLAGS.rnn_layers)
model.build()

# logging.debug(dictionary)


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')
        exit(0)

    for title in titles:
    

        #input = utils.index_data(np.array([[title[0]]]), dictionary)
        input = utils.index_data(np.array([[title[0], title[1], title[2]]]), dictionary)
        state = sess.run(model.initial_state,{model.X: input})
        # feed title
        # for head in title:
            # if head == 0:
                # continue
         
        
            # input = utils.index_data(np.array([[head]]), dictionary)
            
            # feed_dict = {model.X: input,
                         # model.initial_state: state,
                         # model.keep_prob: 1.0}

            # pred, state = sess.run(
                # [model.predictions, model.final_state], feed_dict=feed_dict)
                
            # # print(pred[0][0].argsort()[-1])

        feed_dict = {model.X: input,
                    model.initial_state: state,
                    model.keep_prob: 1.0}    
        pred, state = sess.run(
                    [model.predictions, model.final_state], feed_dict=feed_dict)    
            
        sentence = title
        sentence_i = title
        word_index = pred[0][2].argsort()[-1]
        word = np.take(reverse_list, word_index)
        sentence = sentence + word
        sentence_i = sentence_i + '_[' + str(word_index) + ']_'
        
        logging.debug(word_index)
        
        # logging.debug('============================')
        # logging.debug('input:' + str(input[0][0]) + '-' + str(input[0][1])+ '-' + str(input[0][2]))
        # logging.debug('pred :' + str(pred[0][0].argsort()[-1]) + '-' + str(pred[0][1].argsort()[-1])+ '-' + str(pred[0][2].argsort()[-1]))
        
        
        # generate sample
        for i in range(64):
            #print(np.take(reverse_list, word_index[0]))
            input[0][0] = input[0][1]
            input[0][1] = input[0][2]
            input[0][2] = word_index
            feed_dict = {model.X: input,
                         model.initial_state: state,
                         model.keep_prob: 1.0}

            pred, state = sess.run(
                [model.predictions, model.final_state], feed_dict=feed_dict)

            
            word_index = pred[0][2].argsort()[-1]#pred[0].argsort()[-1]
            # print(word_index)
            
            #logging.debug(len(reverse_list))
            #word_index[0] = max(word_index[0] - 1,0)
            word = np.take(reverse_list, word_index)
            
            # if i < 7:
                # logging.debug('============================')
                # logging.debug('input:' + str(input[0][0]) + '-' + str(input[0][1])+ '-' + str(input[0][2]))
                # logging.debug('pred :' + str(pred[0][0].argsort()[-1]) + '-' + str(pred[0][1].argsort()[-1])+ '-' + str(pred[0][2].argsort()[-1]))
            
            #if word_index[0] <= len(reverse_list) :
            #    word = np.take(reverse_list, word_index[0])
            #else :
            #    word = 'OOB'
            
            
            
            #word = np.take(reverse_list, word_index[0])
            sentence_i = sentence_i + '_[' + str(word_index) + ']_'
            sentence = sentence + word

        logging.debug('==============[{0}]=============='.format(title))
        logging.debug(sentence)
        logging.debug(sentence_i)
