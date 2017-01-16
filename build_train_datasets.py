import json, operator, re, time, pickle
import tensorflow as tf, numpy as np
from vqa_woAtt import *

"""
Feature vector of MSCOCO and list which match image_id to feature vector index
assummed to be built already.
"""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('annotations_path',
            '/data1/shmsw25/vqa/mscoco_train2014_annotations.json',
            """annotation path""")
flags.DEFINE_string('questions_path',
            '/data1/shmsw25/vqa/OpenEnded_mscoco_train2014_questions.json',
            """questions path""")
flags.DEFINE_string('annotations_result_path',
            '/data1/shmsw25/vqa/train_annotations_result',
            """annotation path""")
flags.DEFINE_string('worddic_path',
            '/data1/shmsw25/vqa/',
            """path to save wordtoix and ixtoword""")
flags.DEFINE_string('imgix2featix',
            '/data1/shmsw25/vqa/img2feat',
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
        image_idx_list : list of image ids (with index of feature vector)
        question_list : list of questions (with sentence)
        answer_list : list of answers (with index)
    q_wordtoix
    q_ixtoword
    a_wordtoix
    a_ixtoword
    """
    annotations = json.load(open(FLAGS.annotations_path, 'rb'))
    questions = json.load(open(FLAGS.questions_path, 'rb'))
    image_idx_list, question_list, answer_list = [], [], []

    q_dic, q_word2ix, q_ix2word = {}, {}, {}
    a_dic, a_word2ix, a_ix2word = {}, {}, {}

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
        a_word2ix[a_dic[i][0]] = i
        a_ix2word[i] = a_dic[i][0]

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
    q_word2ix['?'] = 0
    q_ix2word[0] = '?'
    for i in range(1, FLAGS.word_num):
        q_word2ix[q_dic[i][0]] = i
        q_ix2word[i] = q_dic[i][0]

    # (3) create annotations_result
    imgix2featix = pickle.load(open(FLAGS.imgix2featix, 'rb'))
    for dic in annotations:
        q = q_dic[(dic['image_id'], dic['question_id'])]
        for a in dic['answers']:
            if a['answer_confidence'] == 'yes' and a['answer'] in a_word2ix:
                image_idx_list.append(imgix2featix[dic['image_id']])
                question_list.append(q)
                answer_list.append(a_word2ix[a['answer']])

    print "All (img, question, answer) pairs are %d"%(len(image_id_list))
    pickle.dump({'image_idx' : image_idx_list,
        'questions' : question_list,
        'answers' : answer_list},
        open(FLAGS.annotations_result_path, 'wb'))
    print "Success to save Annotation results"
    pickle.dump(q_word2ix, open(FLAGS.worddic_path+'q_word2ix', 'wb'))
    pickle.dump(q_ix2word, open(FLAGS.worddic_path+'q_ix2word', 'wb'))
    pickle.dump(a_word2ix, open(FLAGS.worddic_path+'a_word2ix', 'wb'))
    pickle.dump(a_ix2word, open(FLAGS.worddic_path+'a_ix2word', 'wb'))
    print "Success to save Worddics"

