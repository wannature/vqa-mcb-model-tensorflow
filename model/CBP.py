import tensorflow as tf
import numpy as np
import random

"""
x = [batch_size, origin_dim]
A = [origin_dim, proj_dim]
y = [batch_size, proj_dim]
"""
class CBP():

    def __init__(self, origin_dim, proj_dim):

        self.C = []

        for i in range(2):
            C = np.zeros([origin_dim,proj_dim])
            for i in range(origin_dim):
                C[i][random.randint(0, proj_dim-1)] = 2 * random.randint(0, 1) - 1
            self.C.append(tf.Variable(C, trainable = False, dtype='float32'))

    def bilinear_pool(self, x1, x2):

        p1 = tf.matmul(x1, self.C[0])
        p2 = tf.matmul(x2, self.C[1])
        pc1 = tf.complex(p1, tf.zeros_like(p1))
        pc2 = tf.complex(p2, tf.zeros_like(p2))

        conved = tf.batch_ifft(tf.batch_fft(pc1) * tf.batch_fft(pc2))
        return tf.real(conved)
