from model.mcb_vqamodel import MCB_without_Attention
from model.mcbAtt_vqamodel import MCB_with_Attention
from model.concat_vqamodel import Concat
"""
Use 50000 images and 150000 questions
45000 images and 135000 questions for training
5000 images and 15000 questions for validation
"""

training_img_num = 45000
validation_img_num = 5000

proj_dim = 16000
batch_size = 128
val_batch_size = 8
word_num = 5000
embed_dim = 200
ans_candi_num = 5000
n_lstm_steps = 20
max_epoch = 10

feats_path = '/data1/common_datasets/mscoco/features/train_res_feat.npy'
val_feats_path = '/data1/common_datasets/mscoco/features/val_res_feat.npy'
local_feats_path = '/data1/common_datasets/mscoco/features/train_res4b_feat.npy'
local_val_feats_path = '/data1/common_datasets/mscoco/features/val_res4b_feat.npy'

config_set = {
         'concat' : {'feature_dim' : [2048],
            'feats_path' : feats_path,
            'val_feats_path' : val_feats_path,
            'model' : Concat,
	    'proj_dim' : 4096},
	'mcb' : {'feature_dim' : [2048],
            'feats_path' : feats_path,
            'val_feats_path' : val_feats_path,
            'model' : MCB_without_Attention,
	    'proj_dim' : proj_dim},
        'mcbAtt' : {'feature_dim' : [1024,14,14],
            'feats_path' : local_feats_path,
            'val_feats_path' : local_val_feats_path,
            'model' : MCB_with_Attention,
	    'proj_dim' : proj_dim}
        }

class Config(object):
    def __init__(self, config_name = 'mcb'):

            attset = config_set[config_name]

            self.training_img_num = training_img_num
            self.validation_img_num = validation_img_num
            self.training_num = training_img_num * 3
            self.validation_num = validation_img_num * 3

            self.batch_size = batch_size
            self.val_batch_size = val_batch_size
            self.feature_dim = attset['feature_dim']
            self.proj_dim = attset['proj_dim']
            self.word_num = word_num
            self.embed_dim = embed_dim
            self.ans_candi_num = ans_candi_num
            self.n_lstm_steps = n_lstm_steps
            self.max_epoch = max_epoch

            self.vqamodel = attset['model']

            base_dir = '/data1/shmsw25/vqa/'
            self.model_path = base_dir+'model/'+config_name+'/'
            self.checkpoint = None

            self.annotations_path = base_dir + 'mscoco_train2014_annotations.json'
            self.questions_path = base_dir + \
			'OpenEnded_mscoco_train2014_questions.json'
            self.annotations_result_path = base_dir + 'train_annotations_result'
            self.val_annotations_path = base_dir+'mscoco_val2014_annotations.json'
            self.val_questions_path = base_dir + \
			'OpenEnded_mscoco_val2014_questions.json'
            self.val_annotations_result_path = base_dir + 'val_annotations_result'
            self.selected_val_annotations_path = base_dir+'selected_mscoco_val2014_annotations.json'
            self.selected_val_questions_path = base_dir + \
			'selected_OpenEnded_mscoco_val2014_questions.json'
 

            self.imgix2featix = base_dir+'img2feat'
            self.val_imgix2featix = base_dir+'val_img2feat'
            self.worddic_path = base_dir
            self.feats_path = attset['feats_path']
            self.val_feats_path = attset['val_feats_path']
            self.log_path = base_dir+'log_'+config_name+'/'
            self.result_path = base_dir+'result_'+config_name+'/result-%d'

if __name__ == '__main__':
    config = Config()

