# -*- coding:utf-8 -*-
# Edited by bighead 19-1-31

import tensorflow as tf
import numpy as np
from utils import get_features_labels
from sklearn.preprocessing import  OneHotEncoder

class model:

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate


    def init_weights(self, in_size, out_size, layer_name="empty"):
        with tf.name_scope("weight"):
            weight = tf.Variable(tf.random.uniform([in_size, out_size]),name='weights')
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')

        return (weight, bias)

    def init_inputs(self, data_x, data_y):
        with tf.name_scope("inputs"):
            x = tf.placeholder(tf.float32, [None, len(data_x[0])], name='x')
            y = tf.placeholder(tf.float32, [None, 10], name='y')
        return (x, y)

    def accuracy(self):
        pass


    def loss(self, prediction):
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(prediction + 0.0000001), reduction_indices=[1]))
            tf.summary.scalar('loss', self.cross_entropy)

    def init_Variables(self):
        pass

    def forward(self, x, weight, bias, activation=None):
        if activation is None:
            hidden = tf.matmul(x, weight) + bias
        else:
            hidden = activation(tf.matmul(x, weight) + bias)

        return hidden


    def train(self, data_x, data_y):
        features = len(data_x[0])

        self.x, self.y = self.init_inputs(data_x, data_y)

        ##########forward propagation
        # hidden_unit = 20
        # w1, b1 = self.init_weights(features, hidden_unit, "hidden")
        # hidden = self.forward(x, w1, b1, tf.nn.relu)

        output_unit = 10
        w2, b2 = self.init_weights(features, output_unit, "output")
        prediction = self.forward(self.x, w2, b2, tf.nn.softmax)
        ##########

        self.loss(prediction)

        init = tf.global_variables_initializer()

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        with tf.Session() as sess:
            sess.run(init)
            for i in range(1000):
                sess.run(train_step, feed_dict={self.x:data_x, self.y:data_y})
                if i % 100 == 0:
                    correct_prediction = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(prediction, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print("step: {}, loss: {}, accuracy: {}".format(i, sess.run(self.cross_entropy, feed_dict={self.x:data_x, self.y:data_y}),
                    sess.run(accuracy, feed_dict={self.x:data_x, self.y:data_y})))


if __name__ == "__main__":
    x, y = get_features_labels("data/train.csv")
    enc = OneHotEncoder(sparse=False)
    y = y.reshape(len(y), 1)
    y = enc.fit_transform(y)
    model_ = model()
    model_.train(x, y)
