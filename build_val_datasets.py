import json, operator, re, time, pickle
import tensorflow as tf, numpy as np

training_img_num = 45000
validation_img_num = 5000

# Caffe model : ResNet
res_model = './model/resnet/ResNet-101-model.caffemodel'
res_deploy = './model/resnet/ResNet-101-deploy.prototxt'

layer = {
        'default' : {'layers' : 'res5c_branch2b', 'layer_size' : [2014], 'feat_path' : '/data1/common_datasets/mscoco/features/val_res_feat.npy'},
        '4b' : {'layers' : 'res4b22_branch2c', 'layer_size' : [1024, 14, 14], 'feat_path' : '/data1/common_datasets/mscoco/features/val_res4b_feat.npy'}
        }

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
flags.DEFINE_string('worddic_path',
            '/data1/shmsw25/vqa/',
            """path to save wordtoix and ixtoword""")


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
    """
    annotations = json.load(open(FLAGS.annotations_path, 'rb'))['annotations'][:validation_img_num*3]
    questions = json.load(open(FLAGS.questions_path, 'rb'))['questions'][:validation_img_num*3]
    image_id_list, question_list, question_id_list, answer_list = [], [], [], []

    a_word2ix = pickle.load(open(FLAGS.worddic_path + 'a_word2ix', 'rb'))

    # (1) create annotations_result
    q_dic = {}
    for dic in questions:
        q_dic[(dic['image_id'], dic['question_id'])] = dic['question']

    for dic in annotations:
        q = q_dic[(dic['image_id'], dic['question_id'])]
        image_id_list.append(dic['image_id'])
        question_list.append(q)
        question_id_list.append(dic['question_id'])

    print "All (img, question, answer) pairs are %d"%(len(image_id_list))
    pickle.dump({
        'image_ids' : image_id_list,
        'questions' : question_list},
        open(FLAGS.annotations_result_path, 'wb'))
    print "Success to save Annotation results"

    """
    # (2) Create image features
    # If you run this seperatly, load image_id_list
    # image_id_list = pickle.load(open(FLAGS.annotations_result_path, 'rb'))['image_ids']
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
    """

if __name__ == '__main__' :
    create_annotations_result()

