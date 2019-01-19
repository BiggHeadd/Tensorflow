#-*- coding:utf-8 -*-
#Edited by bighead 19-1-19

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    with tf.name_scope('layer%s' %n_layer):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random.normal([in_size, out_size], name='Weights'))
        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b, keep_prob
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
        

#data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

#input
with tf.name_scope('input'):
    with tf.name_scope('xs'):
        xs = tf.placeholder(tf.float32, [None, 64])
    with tf.name_scope('ys'):
        ys = tf.placeholder(tf.float32, [None, 10])

#param
keep_prob = tf.placeholder(tf.float32)

#neural network
l1 = add_layer(xs, 64, 50, '1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, '2', activation_function=tf.nn.softmax)

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
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    for i in range(500):
        sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        if i % 50 == 0:
            print(i)
            rs = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
            writer.add_summary(rs, i)
