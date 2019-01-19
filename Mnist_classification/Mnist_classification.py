#-*- coding:utf-8 -*-
#Edited by bighead 19-1-19

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_size, out_size, activation_function=None):
    #activation_function(inputs*weights+b)
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random.uniform([in_size, out_size]), name='Weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

def accuracy(x_target, y_target):
    global prediction
    y_predict = sess.run(prediction, feed_dict={xs:x_target})
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:x_target, ys:y_target})
    return result

#mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#x, y placeholder
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='input_x')
    ys = tf.placeholder(tf.float32, [None, 10], name='input_y')

#outputs variable
with tf.name_scope('prediction'):
    prediction = add_layer(xs, 784, 10, tf.nn.softmax)

#loss
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)

#train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#init
init = tf.global_variables_initializer()


#sess
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        x_data, y_data = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            print(accuracy(mnist.test.images, mnist.test.labels))
