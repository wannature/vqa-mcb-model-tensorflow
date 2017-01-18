import json, operator, re, time, pickle
import tensorflow as tf, numpy as np
from vqa_woAtt import *

"""
Feature vector of MSCOCO and list which match image_id to feature vector index
assummed to be built already.
"""

training_img_num = 45000
validation_img_num = 5000

# Caffe model : ResNet
res_model = './model/resnet/ResNet-101-model.caffemodel'
res_deploy = './model/resnet/ResNet-101-deploy.prototxt'

layer = {
        'default' : {layers : 'res5c_branch2b', layer_size : [2014], feat_path = '/data1/common_datasets/mscoco/features/val_res_feat.npy'}
        '4b' : {layers = 'res4b22_branch2c', layer_size=[1024, 14, 14], feat_path = '/data1/common_datasets/mscoco/features/val_res4b_feat.npy'}}

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('annotations_path',
            '/data1/shmsw25/vqa/mscoco_val2014_annotations.json',
            """annotation path""")
flags.DEFINE_string('questions_path',
            '/data1/shmsw25/vqa/OpenEnded_mscoco_val2014_questions.json',
            """questions path""")
flags.DEFINE_string('annotations_result_path',
            '/data1/shmsw25/vqa/val_annotations_result',
            """annotation path""")
flags.DEFINE_string('image_path',
            '/data1/common_datasets/mscoco/images/val2014/',
            """path for validation images""")
flags.DEFINE_string('imgix2featix',
            '/data1/shmsw25/vqa/val_img2feat',
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
        image_id_list : list of image ids (with original index)
        question_list : list of questions (with sentence)
        question_id_list : list of question ids (with index)
        answer_list : list of answers (with index)
    """
    annotations = json.load(open(FLAGS.annotations_path, 'rb'))[validation_img_num*3]
    questions = json.load(open(FLAGS.questions_path, 'rb'))[validation_img_num*3]
    image_id_list, question_list, question_id_list, answer_list = [], [], [], []

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
    for dic in annotations:
        q = q_dic[(dic['image_id'], dic['question_id'])]
        for a in dic['answers']:
            if a['answer_confidence'] == 'yes' and a['answer'] in a_word2ix:
                image_id_list.append(dic['image_id'])
                question_list.append(q)
                question_id_list.append(dic['question_id'])
                answer_list.append(a_word2ix[a['answer']])

    print "All (img, question, answer) pairs are %d"%(len(image_id_list))
    pickle.dump({
        'image_idx' : image_id_list,
        'questions' : question_list,
        'answers' : answer_list},
        open(FLAGS.annotations_result_path, 'wb'))
    print "Success to save Annotation results"


    # (4) Create image features

    unique_image_ids = image_id_list.unique()
    unique_images = unique_image_ids.map(lambda x: \
            os.path.join(image_path+("%12s"%str(x)).replace(" ","0")+".jpeg"))
    print "Unique images are %d" %(len(unique_images))
    for i in range(len(unique_images)):
        imgix2featix[unique_image_ids[i]] = i
    pickle.dump(imgix2featix, open(FLAGS.imgix2featix, 'wb'))

    cnn = CNN(model=res_model, deploy=res_deploy, width=224, height=224)

    for dic in layers:

        layers = dic['layers']
        layer_size = dic['layer_size']
        feat_path = dic['feat_path']
        if not os.path.exists(feat_path):
            feats = cnn.get_features(unique_images,
                layers=layers, layer_sizes=layer_size)
            np.save(FLAGS.feat_path, feats)
    print "Success to save features"


