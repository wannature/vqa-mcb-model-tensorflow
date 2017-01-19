import tensorflow as tf
from vqamodel import *

class MCB_without_Attention(VQAModel):
    def __init__(self, batch_size, feature_dim, proj_dim, \
            word_num, embed_dim, ans_candi_num, n_lstm_steps):

        VQAModel.__init__(self, batch_size, feature_dim, proj_dim, \
                word_num, embed_dim, ans_candi_num, n_lstm_steps)

	self.debug_dic = {}

    def model(self):
        image_feat = tf.placeholder("float32", [self.batch_size, self.feature_dim[0]])
        question = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])
        answer = tf.placeholder("int32", [self.batch_size])

        ques_feat = self.question_embed(question)
	feat = self.bilinear_pool(image_feat, ques_feat)
        signed_feat = tf.sign(feat)*tf.sqrt(tf.abs(feat))
        normalized_feat = tf.nn.l2_normalize(signed_feat, 0)
	logit = tf.matmul(normalized_feat, self.fc_W) + self.fc_b

	self.debug_dic['feat'] = feat
	self.debug_dic['normal_feat'] = normalized_feat
	self.debug_dic['logit'] = logit
	
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
    model = VQA_without_Attention(128, [2048], 16000, 20000, 300, 3000, 50)
    img, q, a, l = model.trainer()
    img, q, a_hat = model.solver()


