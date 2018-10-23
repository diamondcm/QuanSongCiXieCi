#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=3, num_steps=32, num_words=5000, dim_embedding=96, rnn_layers=2):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        batch_size : int
            batch_size.
        num_steps : int
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        #self.dim_embedding = batch_size * num_steps
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate
        
  
        
    def get_init_cell(self,data_shape_0):
        # lstm层数
  

        # dropout时的保留概率
        keep_prob = 0.8

        # 创建包含rnn_size个神经元的lstm cell
        cell = tf.contrib.rnn.BasicLSTMCell(self.dim_embedding,state_is_tuple=True)

        # 使用dropout机制防止overfitting等
        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob,input_keep_prob=1.0)

        # 创建2层lstm层
        cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(self.rnn_layers)], state_is_tuple=True)
        
        # # 使用dropout机制防止overfitting等
        # drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # 初始化状态为0.0
        #print(data_shape_0)
        init_state = cell.zero_state(data_shape_0, tf.float32)

        # 使用tf.identify给init_state取个名字，后面生成文字的时候，要使用这个名字来找到缓存的state
        init_state = tf.identity(init_state, name='init_state')

        return cell, init_state
        
    def build_rnn(self,cell, inputs):
        '''
        cell就是上面get_init_cell创建的cell
        '''

        #print("inputs shape:" + str(inputs.shape))
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        # 同样给final_state一个名字，后面要重新获取缓存
        final_state = tf.identity(final_state, name="final_state")

        return outputs, final_state

    def build_nn(self,cell, embed, vocab_size):

        # 创建embedding layer
        # embed = get_embed(input_data, vocab_size, rnn_size)

        # 计算outputs 和 final_state
        outputs, final_state = self.build_rnn(cell, embed)

        # remember to initialize weights and biases, or the loss will stuck at a very high point
        logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                                   weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                                   biases_initializer=tf.zeros_initializer())

        return logits, final_state


    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)
            
            

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')#
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')#

        #self.LR = tf.placeholder(tf.float32, name='learning_rate')

            
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)
            
            #print("data size:" +str(data.shape))
        
        with tf.variable_scope('rnn'):
            ##################
            # Your Code here
            ##################
            # 创建包含rnn_size个神经元的lstm cell
            
             # 这里的rnn_size表示每个lstm cell中包含了多少的神经元
            cell, self.initial_state = self.get_init_cell(tf.shape(data)[0])

            # 创建计算loss和finalstate的节点
            logits, self.final_state = self.build_nn(cell, data, self.num_words)

            #print(logits.shape)
           
            # self.initial_state = tf.zeros(                                                # initial state of RNN
                # [self.batch_size, self.rnn_layers])
            # state = self.initial_state
            # rnn_outputs = []
            # for current_input in data:                                # tstep 多少个时刻，多少个单词
                
                # scope.reuse_variables()
                # RNN_H = tf.get_variable(
                    # 'HMatrix', [self.rnn_layers, self.rnn_layers])          
                # RNN_I = tf.get_variable(
                    # 'IMatrix', [self.embed_size, self.rnn_layers])
                # RNN_b = tf.get_variable(
                    # 'B', [self.rnn_layers])
                # state = tf.nn.sigmoid(
                    # tf.matmul(state, RNN_H) + tf.matmul(current_input, RNN_I) + RNN_b)      # 这里state是当前时刻的隐藏层
                # rnn_outputs.append(state)                                                   # 不过它在下一个循环中就被用了，所以也是用来存上一时刻隐藏层的
            # self.final_state = rnn_outputs[-1]


        # flatten it
        #seq_output_final = tf.reshape(logits, [-1, self.dim_embedding])

        self.predictions = tf.nn.softmax(logits, name='predictions')
        
        with tf.variable_scope('softmax'):
            ##################
            # Your Code here
            ##################
            ##################
            #logits = tf.nn.softmax(logits, name='probs')
            
            # U = tf.get_variable('Matrix', [self.config.rnn_layers, num_words])
            # proj_b = tf.get_variable('Bias', [num_words])                       
            # logits = [tf.matmul(o, U) + proj_b for o in rnn_outputs]
            pass    
            

        tf.summary.histogram('logits', logits)

        
         # 计算loss
        # loss = seq2seq.sequence_loss(
            # logits,
            # self.Y,
            # tf.ones([tf.shape(data)[0], tf.shape(data)[1]]))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Y, [-1, self.dim_embedding]), logits = logits)
        
        mean,var = tf.nn.moments(logits, -1)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)

        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))
        tf.summary.scalar('var_loss', var_loss)
        # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)
        
        

        self.merged_summary_op = tf.summary.merge_all()
