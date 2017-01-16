import json, operator, re, time
import tensorflow as tf, numpy as np
from vqa_woAtt import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('training_num', 50000, """Num of train questions""")
flags.DEFINE_integer('batch_size', 128, """Batch size""")
flags.DEFINE_integer('feature_dim', 2048, """dimension of feature""")
flags.DEFINE_integer('proj_dim', 16000, """dimension of projection""")
flags.DEFINE_integer('word_num', 20000, """num of words in question""")
flags.DEFINE_integer('embed_dim', 300, """question embedding size""")
flags.DEFINE_integer('ans_candi_num', 3000, """answer size""")
flags.DEFINE_integer('n_lstm_steps', 50, """n_lstm_steps""")
flags.DEFINE_integer('max_epoch', 100, """max epoch to run trainer""")

flags.DEFINE_string('model_path', '/data1/shmsw25/model/vqa/noAtten/',
            """model directory""")
flags.DEFINE_string('checkpoint', None,
            """if sets, resume training on the checkpoint""")

flags.DEFINE_string('annotations_path',
            '/data1/shmsw25/vqa/mscoco_train2014_annotations.json',
            """annotation path""")
flags.DEFINE_string('questions_path',
            '/data1/shmsw25/vqa/OpenEnded_mscoco_train2014_questions.json',
            """questions path""")
flags.DEFINE_string('annotations_result_path',
            '/data1/shmsw25/vqa/train_annotations_result.json',
            """annotation path""")
flags.DEFINE_string('worddic_path',
            '/data1/shmsw25/vqa/',
            """path to save wordtoix and ixtoword""")
flags.DEFINE_string('feats_path',
            '/data1/shmsw25/vqa/train_res_feats.npy',
            """features of images path""")

def train():
    learning_rate = 0.001

    sess = Session()
    model = VQA_without_Attention(
            batch_size=FLAGS.batch_size,
            feature_dim=FLAGS.feature_dim,
            proj_dim=FLAGS.proj_dim,
            word_num=FLAGS.word_num,
            embed_dim=FLAGS.embed_dim,
            ans_candi_num=FLAGS.ans_candi_num,
            n_lstm_steps=FLAGS.n_lstm_steps)

    image_feat, question, answer, loss = model.model()
    saver = tf.train.Saver(max_to_keep = 50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess.run(tf.initialize_all_variables())
    if FLAGS.checkpoint:
        saver.restore(sess, FLAGS.model_path+"-%d"%(checkpoint))

    from_idx = range(0, FLAGS.training_num, FLAGS.batch_size)
    to_idx = range(FLAGS.batch_size, FLAGS.training_num, FLAGS.batch_size)

    annotations_result = json.load(open(FLAGS.annotations_result_path, 'rb'))
    image_idx = annotations_result['image_idx']
    questions = annotations_result['questions']
    answers = annotations_result['answers']
    q_word2ix = json.load(open(FLAGS.worddic_path+'q_word2ix', 'rb'))
    feats = np.load(FLAGS.feats_path)

    print "*** Training Start ***"

    for epoch in range(FLAGS.max_epoch):
        print "Start running epoch %d" % (epoch)
        t = time.time()

        shuffler = np.random.permutation(FLAGS.training_num)
        image_ids = image_ids[shuffler]
        questions = questions[shuffler]
        answers = answers[shuffler]

        for (start, end) in zip(from_idx, to_idx):
            # make curr_image_feat [batch_size, feature_dim]
            curr_image_feat = feats[image_idx[start:end]]
            # make curr_question [batch_size, n_lstm_steps]
            curr_question = map(lambda ques :
                [q_word2ix[word] for word in ques.lower() if word in q_word2ix],
                questions[start:end])
            curr_question = sequence.pad_sequences(
                    curr_question, padding='post', maxlen=FLAGS.n_lstm_steps)
            # make curr_answer [batch_size]
            curr_answer = answers[start:end]

            _, loss = sess.run([train_op, loss],
                    feed_dict = {image_feat : curr_image_feat,
                                question : curr_question,
                                answer : curr_answer})

        print "End running epoch %d : %dmin" %(epoch, (time.time()-t)/60)
        saver.save(sess, os.path.join(FLAGS.model_path, 'model'),
                global_step = epoch)



