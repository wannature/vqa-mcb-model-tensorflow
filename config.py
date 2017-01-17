from vqa_woAtt import VQA_without_Attention
from vqa_Att import VQA_with_Attention

attention_set = {
        0 : {'name' : 'noAtten/', 'feature_dim' : [2048],
            'feats_path' = 'train_res_feats.npy',
            'val_feats_path' = 'val_res_feats.npy',
            'model' : VQA_without_Attention},
        1 : {'name' : 'Atten1/', 'feature_dim' : [1024,14,14],
            'feats_path' : 'train_res4b_feats.npy',
            'val_feats_path' = 'val_res4b_feats.npy',
            'model' : VQA_with_Attention}}

class Config(object):
    def __init__(self, attention_num = 0, proj_dim = 16000,
            training_num = 50000, validation_num = 5000,
            batch_size = 128, val_batch_size = 32, word_num = 20000, embed_dim = 300,
            ans_candi_num = 3000, n_lstm_steps = 50, max_epoch = 100):

            attset = attention_set[attention_num]
            self.training_num = training_num
            self.batch_size = batch_size
            self.val_batch_size = val_batch_size
            self.feature_dim = attset['feature_dim']
            self.proj_dim = proj_dim
            self.word_num = word_num
            self.embed_dim = embed_dim
            self.ans_candi_num = ans_candi_num
            self.n_lstm_steps = n_lstm_steps
            self.max_epoch = max_epoch

            self.vqamodel = attset['model']

            base_dir = '/data1/shmsw25/vqa/'
            self.model_path = base_dir+'model/'+attset['name']
            self.checkpoint = None

            self.annotations_path = base_dir+'mscoco_train2014_annotations.json'
            self.questions_path = base_dir+'OpenEnded_mscoco_train2014_questions.json'
            self.annotations_result_path = base_dir+'train_annotations_result.json'
            self.val_annotations_path = base_dir+'mscoco_val2014_annotations.json'
            self.val_questions_path = base_dir+'OpenEnded_mscoco_val2014_questions.json'
            self.val_annotations_result_path = base_dir+'val_annotations_result.json'

            self.worddic_path = base_dir
            self.feats_path = base_dir+attset['feats_path']
            self.val_feats_path = base_dir+attset['val_feats_path']
            self.result_path = base_dir+'result_'+attset['name']+'result-%d'



