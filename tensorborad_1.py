#-*- coding:utf-8 -*-
#Edited by bighead 19-1-19

import tensorflow as tf
import numpy as np

#a neural network layer
def add_layer(inputs, in_size, out_size, n_layer, activation_function):
    layer_name = 'layers%d' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('b'):
            biases = tf.Variable(tf.zeros([1, out_size]) +0.1)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            tf.summary.histogram(layer_name + '/Wx_plus_b', Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs

#input values
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_inputs')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_inputs')

#build neural network
lay1 = add_layer(xs, 1, 10, 1, tf.nn.relu)
prediction = add_layer(lay1, 10, 1, 2, None)

#loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
#train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#Session
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        rs = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(rs, i)
sess.close()
