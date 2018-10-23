#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import numpy as np

import tensorflow as tf

import utils
from model import Model
from utils import read_data
from utils import build_dataset
from flags import train_times

from flags import parse_args
FLAGS, unparsed = parse_args()

#print('current working dir [{0}]'.format(os.getcwd()))
w_d = os.path.dirname(os.path.abspath(__file__))
#print('change wording dir to [{0}]'.format(w_d))
os.chdir(w_d)

cmd = ""
for parm in ["output_dir", "text", "num_steps", "batch_size", "dictionary", "reverse_dictionary", "learning_rate"]:
    try:
        cmd += ' --{0}={1}'.format(parm, getattr(FLAGS, parm))
    except:
        pass


#print("##########################new start################################")
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))


# with open(FLAGS.dictionary, encoding='utf-8') as inf:
    # dictionary = json.load(inf, encoding='utf-8')

# with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    # reverse_dictionary = json.load(inf, encoding='utf-8')

# data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, 10000)
# data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, len(count))    
# reverse_list = [reverse_dictionary[i]
                # for i in range(len(reverse_dictionary))]  

   
batches,vol_len = utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, seq_length=FLAGS.num_steps)
    
logging.debug(vol_len)
model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps, num_words = vol_len,rnn_layers = FLAGS.rnn_layers)
model.build()



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

    ePerT = 50    
    hasShownFirst = False
    for e in range(ePerT):
        #logging.debug('epoch [{0}]....'.format(e))
        #print(batches[0][0])
        state = sess.run(model.initial_state, {model.X:batches[0][0]})
        for batch_i, (x,y) in enumerate(batches):
            #logging.debug("batch_i:" + str(batch_i))
            # print("XXXXXXXXXXXXXXX:" + str(x))
            # print("YYYYYYYYYYYYYYY:" + str(y))
            # wordx = ''
            # for i in range(len(x[0])):
                # wordx = wordx + np.take(reverse_list, str(x[0][i]))
            # print(wordx)
            
            # wordy = ''
            # for i in range(len(y[0])):
                # wordy = wordy + np.take(reverse_list, str(y[0][i]))
            # print(wordy)
            
            feed_dict = {
                model.X:x,
                model.Y:y,
                model.initial_state:state,
                #model.LR:FLAGS.learning_rate
                }
            ##################
            # Your Code here
            ##################

            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.final_state, model.loss, model.merged_summary_op], feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            per = format(batch_i / len(batches),'.0%')
            if gs % 100 == 0:
                logging.debug( per+' step [{0}] loss [{1}]'.format(gs, l) + "\r")
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
                    
            if gs % 5000 == 0 or ( not hasShownFirst):
                hasShownFirst = True
                print('################    eval    ################')
                p = os.popen('python ./sample.py' + cmd)
                for l in p:
                    print(l.strip())
        logging.debug( 'epoch [{2}] finished. step [{0}] loss [{1}]'.format(gs, l,e + train_times * ePerT))
        save_path = saver.save(sess, os.path.join(
            FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
