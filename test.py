import json, operator, re, time
import tensorflow as tf, numpy as np
from vqa_woAtt import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('valid_num', 5000, """Num of train questions""")
flags.DEFINE_integer('batch_size', 128, """Batch size""")
flags.DEFINE_integer('feature_dim', 2048, """dimension of feature""")
flags.DEFINE_integer('proj_dim', 16000, """dimension of projection""")
flags.DEFINE_integer('word_num', 20000, """num of words in question""")
flags.DEFINE_integer('embed_dim', 300, """question embedding size""")
flags.DEFINE_integer('ans_candi_num', 3000, """answer size""")
flags.DEFINE_integer('n_lstm_steps', 50, """n_lstm_steps""")

flags.DEFINE_string('model_path', '/data1/shmsw25/model/vqa/noAtten/model-%d',
            """model directory""")
flags.DEFINE_string('annotations_path',
            '/data1/shmsw25/vqa/mscoco_val2014_annotations.json',
            """annotation path""")
flags.DEFINE_string('questions_path',
            '/data1/shmsw25/vqa/OpenEnded_mscoco_val2014_questions.json',
            """questions path""")
flags.DEFINE_string('annotations_result_path',
            '/data1/shmsw25/vqa/val_annotations_result.json',
            """annotation path""")
flags.DEFINE_string('worddic_path',
            '/data1/shmsw25/vqa/',
            """path to save wordtoix and ixtoword""")
flags.DEFINE_string('feats_path',
            '/data1/shmsw25/vqa/val_res_feats.npy',
            """features of images path""")
flags.DEFINE_string('result_path',
            '/data1/shmsw25/vqa/result_noAtten/result-%d',
            """path for result""")


def train(epoch):

    sess = Session()
    model = VQA_without_Attention(
            batch_size=FLAGS.batch_size,
            feature_dim=FLAGS.feature_dim,
            proj_dim=FLAGS.proj_dim,
            word_num=FLAGS.word_num,
            embed_dim=FLAGS.embed_dim,
            ans_candi_num=FLAGS.ans_candi_num,
            n_lstm_steps=FLAGS.n_lstm_steps)

    image_feat, question, max_prob_words = model.solver()
    saver = tf.train.Saver(max_to_keep = 50)


    saver.restore(sess, FLAGS.model_path%(epoch))

    from_idx = range(0, FLAGS.valid_num, FLAGS.batch_size)
    to_idx = range(FLAGS.batch_size, FLAGS.valid_num, FLAGS.batch_size)

    annotations_result = json.load(open(FLAGS.annotations_result_path, 'rb'))
    image_idx = annotations_result['image_idx']
    image_ids = annotations_result['image_ids']
    questions = annotations_result['questions']
    question_ids = annotations_result['question_ids']
    answers = annotations_result['answers']
    q_word2ix = pickle.load(open(FLAGS.worddic_path+'q_word2ix', 'rb'))
    a_ix2word = pickle.load(open(FLAGS.worddic_path+'a_ix2word', 'rb'))
    feats = np.load(FLAGS.feats_path)

    print "*** Test Start ***"

    result = []
    for (start, end) in zip(from_idx, to_idx):
        # make curr_image_feat [batch_size, feature_dim]
        curr_image_feat = feats[start:end]
        # make curr_question [batch_size, n_lstm_steps]
        curr_question = map(lambda ques :
            [q_word2ix[word] for word in ques.lower() if word in q_word2ix],
            questions[start:end])
        curr_question = sequence.pad_sequences(
                curr_question, padding='post', maxlen=FLAGS.n_lstm_steps)

        answer_ids = sess.run(max_prob_words,
                feed_dict = {image_feat : curr_image_feat,
                            question : curr_question})

        answers = map(lambda ix : a_ix2word[ix], answer_ids)
        for i in range(FLAGS.batch_size):
            result.append({
                'question_id' : question_ids[start+i],
                'answer' : answers[i]
                })

    pickle.dump(result, FLAGS.result_path%(epoch))
