#-*- coding:utf-8 -*-
#Edited by bighead 19-1-16

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

result = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(result, feed_dict={input1:[7.], input2:[2.]}))
