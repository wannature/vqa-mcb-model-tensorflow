import tensorflow as tf
from vqamodel import VQAModel

class VQA_without_Attention(VQAModel):
    def __init__(self, batch_size, feature_dim, proj_dim,
            word_num, embed_dim, ans_candi_num, n_lstm_steps):
        super(VQAModel, self).__init__(batch_size, feature_dim, proj_dim,
            word_num, embed_dim, ans_candi_num, n_lstm_steps)

    def model(self):
        image_feat = tf.placeholder("float32", [self.batch_size, self.feature_dim])
        question = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])
        answer = tf.placeholder("int32", [self.batch_size])

        ques_feat = self.question_embed(question)
        feat = bilinear_pool(image_feat, ques_feat, self.proj_dim)

        signed_feat = tf.sign(feat)*tf.sqrt(feat)
        normalized_feat = tf.nn.l2_normalize(signed_feat, 0)
        logit = tf.matmul(normalized_feat, self.fc_W) + self.fc_b
        loss = self.get_loss(logit, answer)

        return image_feat, question, answer, loss

    def solver(self):
        image_feat = tf.placeholder("float32", [self.batch_size, self.feature_dim])
        question = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])

        ques_feat = self.question_embed(question)
        feat = bilinear_pool(image_feat, ques_feat, self.proj_dim)

        signed_feat = tf.sign(feat)*tf.sqrt(feat)
        normalized_feat = tf.nn.l2_normalize(signed_feat, 0)
        logit = tf.matmul(normalized_feat, self.fc_W) + self.fc_b
        max_prob_words = tf.argmax(logit, 1)

        return image_feat, question, max_prob_words







