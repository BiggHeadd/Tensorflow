#-*- coding:utf-8 -*-
#edited by bighead 19-1-16

import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
