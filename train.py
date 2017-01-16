import json, operator, re, time
import tensorflow as tf, numpy as np
from qa_model import *

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

def create_annotations_result():
    """
    annotations['annotations']
        answers : list of 10 answers (answer, answer_confidence, answer_id)
        image_id
        question_id
    questions['questions']
        image_id
        question
        question_id
    annotations_result(train only answers with confidence 'yes')
        image_id_list : list of image ids (with index)
        question_list : list of questions (with sentence)
        answer_list : list of answers (with index)
    q_wordtoix
    q_ixtoword
    a_wordtoix
    a_ixtoword
    """
    annotations = json.load(open(FLAGS.annotations_path, 'rb'))
    questions = json.load(open(FLAGS.questions_path, 'rb'))
    image_id_list, question_list, answer_list = [], [], []

    q_dic, q_wordtoix, q_ixtoword = {}, {}, {}
    a_dic, a_wordtoix, a_ixtoword = {}, {}, {}

    # (1) create wordtoix, ixtoword for answers
    for dic in annotations:
        for a in dic['answers']:
            if a['answer_confidence'] == 'yes':
                ans = a['answer']
                if ans in a_dic: a_dic[ans] += 1
                else: a_dic[ans] = 1
    a_dic = sorted(a_dic.items(), key=operator.itemgetter(1))
    a_dic.reverse()
    for i in range(FLAGS.ans_candi_num):
        a_wordtoix[a_dic[i][0]] = i
        a_ixtoword[i] = a_dic[i][0]

    # (2) create wordtoix, ixtoword for questions
    p = re.compile('[\w]')
    q_len_dic = {}
    for dic in questions:
        q_dic[(dic['image_id'], dic['question_id'])] = dic['question']
        q_words = p.findall(dic['question'].lower())
        for qw in q_words:
            if qw in q_dic : q_dic[qw] += 1
            else: q_dic[qw] = 1
        q_len_key = 10*int(len(q_words)/10)
        if q_len_key in q_len_dic : q_len_dic[q_len_key] += 1
        else : q_len_dic[q_len_key] = 1
    print "Dictionary about the length of questions"
    for q_len_key in q_len_dic:
        "%d ~ %d\t: %d" %(q_len_key, q_len_key+10, q_len_dic[q_len_key])
    q_dic = sorted(q_dic.items(), key=operator.itemgetter(1))
    q_dic.reverse()
    q_wordtoix['?'] = 0
    q_ixtoword[0] = '?'
    for i in range(1, FLAGS.word_num):
        q_wordtoix[q_dic[i][0]] = i
        q_ixtoword[i] = q_dic[i][0]

    # (3) create annotations_result
    for dic in annotations:
        q = q_dic[(dic['image_id'], dic['question_id'])]
        for a in dic['answers']:
            if a['answer_confidence'] == 'yes' and a['answer'] in a_wordtoix:
                image_id_list.append(dic['image_id'])
                question_list.append(q)
                answer_list.append(a_wordtoix[a['answer']])

    print "All (img, question, answer) pairs are %d"%(len(image_id_list))
    json.dump({'image_ids' : image_id_list,
        'questions' : question_list,
        'answers' : answer_list},
        open(FLAGS.annotations_result_path, 'wb'))
    print "Success to save Annotation results"
    json.dump(q_wordtoix, open(FLAGS.worddic_path+'q_wordtoix', 'wb'))
    json.dump(q_ixtoword, open(FLAGS.worddic_path+'q_ixtoword', 'wb'))
    json.dump(a_wordtoix, open(FLAGS.worddic_path+'a_wordtoix', 'wb'))
    json.dump(a_ixtoword, open(FLAGS.worddic_path+'a_ixtoword', 'wb'))
    print "Success to save Worddics"


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
    image_ids = annotations_result['image_ids']
    questions = annotations_result['questions']
    answers = annotations_result['answers']
    q_wordtoix = json.load(open(FLAGS.worddic_path+'q_wordtoix', 'rb'))

    for epoch in range(FLAGS.max_epoch):
        print "Start running epoch %d" % (epoch)
        t = time.time()
        for (start, end) in zip(from_idx, to_idx):
            # make curr_image_feat [batch_size, feature_dim]
            # make curr_question [batch_size, n_lstm_steps]
            curr_question = map(lambda ques :
                [q_wordtoix[word] for word in ques.lower() if word in q_wordtoix],
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



