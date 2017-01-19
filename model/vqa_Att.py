import tensorflow as tf
from vqamodel import *
from CBP import CBP

class VQA_with_Attention(VQAModel):
    def __init__(self, batch_size, feature_dim, proj_dim,
            word_num, embed_dim, ans_candi_num, n_lstm_steps):
        VQAModel.__init__(self,batch_size, feature_dim, proj_dim,
            word_num, embed_dim, ans_candi_num, n_lstm_steps)

        self.local_num = self.feature_dim[1]*self.feature_dim[2]
        cbp_local = CBP(self.feature_dim[0], self.proj_dim)
        self.bilinear_pool_local = cbp_local.bilinear_pool

    def conv_forward_prop(self, input, shape, strides, alpha=0.1):
        kernel = self.init_weight(shape)
        conv = tf.nn.conv2d(input,kernel,strides, padding='SAME')
        biases = self.init_bias(shape[-1])
        return tf.nn.bias_add(conv, biases)

    def model(self):

        image_feat = tf.placeholder("float32", [self.batch_size,
            self.feature_dim[0], self.feature_dim[1], self.feature_dim[2]])
        question = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])
        answer = tf.placeholder("int32", [self.batch_size])

        # ready for first MCB module
        reshaped_image_feat = tf.reshape(image_feat, [self.batch_size,
                self.feature_dim[0], self.feature_dim[1]*self.feature_dim[2]])
        ques_feat = self.question_embed(question)

        tiled_ques_feat = tf.tile(tf.reshape(ques_feat, [self.batch_size*self.feature_dim[0]]),
                [self.local_num])
        tiled_ques_feat = tf.reshape(tiled_ques_feat, [self.local_num, self.batch_size, self.feature_dim[0]])
        tiled_ques_feat = tf.transpose(tiled_ques_feat, [1, 2, 0])

        # Check if reshaped_image_feat and tied_ques_feat have proper shape
        #check_size = [self.batch_size, self.feature_dim[0], self.local_num]
        #check_shape(reshaped_image_feat, 'img_before_mcb1', check_size)
        #check_shape(tiled_ques_feat, 'ques_before_mcb2', check_size)

        # First MCB module
        before_pool_image_feat = tf.reshape(
                tf.transpose(reshaped_image_feat, [0, 2, 1]),
                [self.batch_size * self.local_num, self.feature_dim[0]])
        before_pool_ques_feat = tf.reshape(
                tf.transpose(tiled_ques_feat, [0, 2, 1]),
                [self.batch_size * self.local_num, self.feature_dim[0]])
        att = self.bilinear_pool_local(before_pool_image_feat, before_pool_ques_feat)
        att = tf.reshape(att, [self.batch_size, self.local_num, self.proj_dim])
        att = tf.reshape(tf.transpose(att, [0, 2, 1]),
                [self.batch_size, self.proj_dim,
                    self.feature_dim[1], self.feature_dim[2]])

        # end for First MCB module
        signed_att = tf.sign(att)*tf.sqrt(att)
        normalized_att = tf.nn.l2_normalize(signed_att, 0)
        normalized_att = tf.transpose(normalized_att, [0, 2, 3, 1])

        conv1 = self.conv_forward_prop(normalized_att,
            [3, 3, self.proj_dim, 512],
            [1, 1, 1, 1])
        out1 = tf.maximum(conv1, 0.01*conv1)
        conv2 = self.conv_forward_prop(out1,
            [3, 3, 512, 1],
            [1, 1, 1, 1])
        out2 = tf.reshape(conv2, [self.batch_size, self.local_num])
        alpha = tf.nn.softmax(out2)
        alpha = tf.reshape(alpha, [self.batch_size, self.local_num])
        att_feat = tf.reduce_sum(
                tf.transpose(tf.reshape(image_feat,
                    [self.batch_size, self.feature_dim[0], self.local_num]),
                    [0, 2, 1]) * \
                    tf.expand_dims(alpha, 2), 1)

        # Check if ques_feat and att_feat have proper shape
        #check_size = [self.batch_size, self.feature_dim[0]]
        #check_shape(att_feat, 'att before mcb2', check_size)
        #check_shape(ques_feat, 'ques before mcb2', check_size)

        # Second MCB module
        feat = self.bilinear_pool(ques_feat, att_feat)

        signed_feat = tf.sign(feat)*tf.sqrt(feat)
        normalized_feat = tf.nn.l2_normalize(signed_feat, 0)
        logit = tf.matmul(normalized_feat, self.fc_W) + self.fc_b

        return image_feat, question, answer, logit

    def trainer(self):

        image_feat, question, answer, logit = self.model()
        loss = self.get_loss(logit, answer)
        return image_feat, question, answer, loss

    def solver(self):

        image_feat, question, answer, logit = self.model()
        max_prob_words = tf.argmax(logit, 1)

        return image_feat, question, max_prob_words

if __name__ == '__main__':
    model = VQA_with_Attention(128, [1024,14,14], 16000, 20000, 300, 3000, 50)
    img, q, a, l = model.trainer()
    img, q, a_hat = model.solver()


