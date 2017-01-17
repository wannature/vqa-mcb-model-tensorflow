import tensorflow as tf
import math, numpy as np
# compact bilinear pooling is cloned from
#       https://github.com/therne/compact-bilinear-pooling-tf
#from CBP.count_sketch import bilinear_pool

def bilinear_pool(x1, x2, proj_dim):
    return tf.zeros([128, proj_dim])

def check_shape(tensor, name, shape):
    _shape = list(tensor.get_shape())
    answer = _shape == shape
    print "Does %s has shape %s?    [%s]" % (name, shape, answer)
    if answer is False:
        print "Its shape is %s" % (_shape)

class VQAModel():
    def __init__(self, batch_size, feature_dim, proj_dim, \
            word_num, embed_dim, ans_candi_num, n_lstm_steps):

        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.lstm_hidden_dim = feature_dim[0]/2
        self.proj_dim = proj_dim
        self.word_num = word_num
        self.embed_dim = embed_dim
        self.ans_candi_num = ans_candi_num
        self.n_lstm_steps = n_lstm_steps

        # Word Embedding E (K*m)
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform(
                [self.word_num, self.embed_dim], -1.0, 1.0), name='Wemb')
        self.init_hidden_W = self.init_weight(
                [self.embed_dim, self.lstm_hidden_dim], name='init_hidden_W')
        self.init_hidden_b = self.init_bias([self.lstm_hidden_dim],
                name='init_hidden_b')
        self.init_memory_W = self.init_weight(
                [self.embed_dim, self.lstm_hidden_dim], name='init_memory_W')
        self.init_memory_b = self.init_bias([self.lstm_hidden_dim],
                name='init_memory_b')

        self.lstm_W1 = self.init_weight(
                [self.embed_dim, self.lstm_hidden_dim*4],name='lstm_W1')
        self.lstm_W2 = self.init_weight(
                [self.lstm_hidden_dim, self.lstm_hidden_dim*4],name='lstm_W2')
        self.lstm_U1 = self.init_weight(
                [self.lstm_hidden_dim, self.lstm_hidden_dim*4], name='lstm_U1')
        self.lstm_U2 = self.init_weight(
                [self.lstm_hidden_dim, self.lstm_hidden_dim*4],name='lstm_U2')
        self.lstm_b1 = self.init_bias([self.lstm_hidden_dim*4],
                name='lstm_b1')
        self.lstm_b2 = self.init_bias([self.lstm_hidden_dim*4],
                name='lstm_b2')

        self.fc_W = self.init_weight([self.proj_dim, self.ans_candi_num],
                name='fc_W')
        self.fc_b = self.init_bias([self.ans_candi_num], name='fc_b')


    # http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    def bn(self, x, scope, is_train=True, reuse=True):
        return tf.contrib.layers.batch_norm(
            x, center=False, scale=True, reuse=bool(reuse),
            scope=scope, is_training=is_train)

    def init_weight(self, shape, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal(shape,
                stddev=stddev/math.sqrt(float(shape[0]))), name=name)

    def init_bias(self, shape, name=None):
        return tf.Variable(tf.zeros(shape), name=name)

    def get_initial_lstm(self, mean_context):
        return tf.zeros([self.batch_size, self.lstm_hidden_dim]), \
                tf.zeros([self.batch_size, self.lstm_hidden_dim])
        """
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)
        return initial_hidden, initial_memory
        """

    def forward_prop(self, h1, c1, h2, c2, word_emb, is_train=True, reuse=False):
        lstm_preactive1 = tf.matmul(h1, self.lstm_U1) + \
                            tf.matmul(word_emb, self.lstm_W1) + self.lstm_b1
        i1, f1, o1, g1 = tf.split(1, 4, lstm_preactive1)
        i1 = tf.nn.sigmoid(i1)
        f1 = tf.nn.sigmoid(f1)
        o1 = tf.nn.sigmoid(o1)
        g1 = tf.nn.tanh(g1)
        c1 = f1*c1 + i1*g1
        h1 = o1*tf.nn.tanh(c1)

        lstm_preactive2 = tf.matmul(h2, self.lstm_U2) + \
                            tf.matmul(h1, self.lstm_W2) + self.lstm_b2
        i2, f2, o2, g2 = tf.split(1, 4, lstm_preactive2)
        i2 = tf.nn.sigmoid(i2)
        f2 = tf.nn.sigmoid(f2)
        o2 = tf.nn.sigmoid(o2)
        g2 = tf.nn.tanh(g2)
        c2 = f2*c2 + i2*g2
        h2 = o2*tf.nn.tanh(c2)

        return h1, c1, h2, c2

    def question_embed(self, question):
        h1, c1 = self.get_initial_lstm(tf.reduce_mean(question, 1))
        h2, c2 = self.get_initial_lstm(tf.reduce_mean(question, 1))

        for idx in range(self.n_lstm_steps):
            if idx == 0:
                word_embed = tf.zeros([self.batch_size, self.embed_dim])
            else:
                tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"):
                    word_embed = tf.nn.embedding_lookup(self.Wemb, question[:,idx-1])
            h1, c1, h2, c2 = self.forward_prop(h1, c1, h2, c2, word_embed, reuse=idx)
        return tf.concat(1, [h1, h2])

    def get_loss(self, logit, answer):
        label = tf.expand_dims(answer, 1)
        index = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
        concated = tf.concat(1, [index, label])
        onehot_label = tf.sparse_to_dense(concated,
                tf.pack([self.batch_size, self.ans_candi_num]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit, onehot_label)
        return tf.reduce_sum(cross_entropy)/self.batch_size







