import json, operator, re, time, pickle, os
from cnn import *

training_img_num = 45000
validation_img_num = 5000

# Caffe model : ResNet
res_model = '/home/shmsw25/caffe/models/resnet/ResNet-101-model.caffemodel'
res_deploy = '/home/shmsw25/caffe/models/resnet/ResNet-101-deploy.prototxt'

layer_set = {
        'default' : {'layers' : 'pool5', 'layer_size' : [2048], 'feat_path' : '/data1/common_datasets/mscoco/features/val_res_feat.npy'},
        '4b' : {'layers' : 'res4b22_branch2c', 'layer_size' : [1024, 14, 14], 'feat_path' : '/data1/common_datasets/mscoco/features/val_res4b_feat.npy'}
        }

annotations_path = '/data1/shmsw25/vqa/mscoco_val2014_annotations.json'
questions_path = '/data1/shmsw25/vqa/OpenEnded_mscoco_val2014_questions.json'
annotations_result_path = '/data1/shmsw25/vqa/val_annotations_result'
image_path = '/data1/common_datasets/mscoco/images/val2014/COCO_val2014_'
imgix2featix_path = '/data1/shmsw25/vqa/val_img2feat'
worddic_path = '/data1/shmsw25/vqa/'


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
    annotations = json.load(open(annotations_path, 'rb'))['annotations'][:validation_img_num*3]
    questions = json.load(open(questions_path, 'rb'))['questions'][:validation_img_num*3]
    image_id_list, question_list, question_id_list, answer_list = [], [], [], []

    a_word2ix = pickle.load(open(worddic_path + 'a_word2ix', 'rb'))

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
        'questions' : question_list,
	'question_ids' : question_id_list},
        open(annotations_result_path, 'wb'))
    print "Success to save Annotation results"
    
    
    # (2) Create image features
    # If you run this seperatly, load image_id_list
    # image_id_list = pickle.load(open(annotations_result_path, 'rb'))['image_ids']
    unique_image_ids = list(set(image_id_list))
    unique_images = map(lambda x: \
            os.path.join(image_path+("%12s"%str(x)).replace(" ","0")+".jpg"),
	    unique_image_ids)
    print "Unique images are %d" %(len(unique_images))
    imgix2featix = {}
    for i in range(len(unique_images)):
        imgix2featix[unique_image_ids[i]] = i
    pickle.dump(imgix2featix, open(imgix2featix_path, 'wb'))

    cnn = CNN(model=res_model, deploy=res_deploy, width=224, height=224)

    for dic in layer_set.values():

        layers = dic['layers']
        layer_size = dic['layer_size']
        feat_path = dic['feat_path']
        if not os.path.exists(feat_path):
            feats = cnn.get_features(unique_images,
                layers=layers, layer_sizes=layer_size)
            np.save(feat_path, feats)
    print "Success to save features"
    

if __name__ == '__main__' :
    create_annotations_result()

