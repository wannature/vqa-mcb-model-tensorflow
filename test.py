import json, operator, re, time, pickle
import tensorflow as tf, numpy as np
from tensorflow.python.framework import ops
from config import Config
from keras.preprocessing import sequence

def test(config=Config(), epoch_list = range(10)):

    from_idx = range(0, config.validation_num, config.val_batch_size)
    to_idx = range(config.val_batch_size, config.val_batch_size+config.validation_num, config.val_batch_size)

    annotations_result = pickle.load(open(config.val_annotations_result_path, 'rb'))
    image_ids = annotations_result['image_ids']
    questions = annotations_result['questions']
    if config.dataset_name == 'mscoco':
        question_ids = annotations_result['question_ids']
    if config.dataset_name == 'genome':
        real_answers = annotations_result['answers']

    q_word2ix = pickle.load(open(config.worddic_path+'q_word2ix', 'rb'))
    a_ix2word = pickle.load(open(config.worddic_path+'a_ix2word', 'rb'))
    imgix2featix = pickle.load(open(config.val_imgix2featix, 'rb'))
    p = re.compile('[\w]+')
    feats = np.load(config.val_feats_path)
    feats = feats[[imgix2featix[imgix] for imgix in image_ids]]


    def test_single(epoch):
        accuracy = 0.0
        tot = 0
        correct = {}
        wrong = {}

        print "*** Test Start for Epoch %d ***" %(epoch)
        sess = tf.Session()
        model = config.vqamodel(
            batch_size=config.val_batch_size,
            feature_dim=config.feature_dim,
            proj_dim=config.proj_dim,
            word_num=config.word_num,
            embed_dim=config.embed_dim,
            ans_candi_num=config.ans_candi_num,
            n_lstm_steps=config.n_lstm_steps)

        sess.run(tf.initialize_all_variables())
	image_feat, question, max_prob_words = model.solver()
        saver = tf.train.Saver(max_to_keep = 50)
        saver.restore(sess, config.model_path+'model-%d'%(epoch))

        result = []
        for (start, end) in zip(from_idx, to_idx):
            # make curr_image_feat [batch_size, feature_dim]
            curr_image_feat = feats[start:end]
            # make curr_question [batch_size, n_lstm_steps]
            curr_question = questions[start:end]
            curr_question = map(lambda ques :
                [q_word2ix[word] for word in p.findall(ques.lower())
                    if word in q_word2ix],
                curr_question)
            curr_question = np.array(sequence.pad_sequences(
                curr_question, padding='post', maxlen=config.n_lstm_steps))

            answer_ids = sess.run(max_prob_words,
                feed_dict = {image_feat : curr_image_feat,
                            question : curr_question})

            answers = map(lambda ix : a_ix2word[ix], answer_ids)

            if config.dataset_name == 'mscoco':
                for i in range(config.val_batch_size):
                    result.append({
                        'question_id' : question_ids[start+i],
                        'answer' : answers[i]
                        })

            if start%(config.val_batch_size*500)==0:
                print questions[start], answers[0]

            # TODO: accuracy evaluation for genome dataset


            if config.dataset_name == 'genome':
                for i in range(config.val_batch_size):
                    a = answers[i]
                    real_a = real_answers[start+i]
                    real_a = p.findall(real_a.lower())[0]
                    tot += 1
                    if a == real_a:
                        accuracy += 1
                        if a in correct: correct[a] += 1
                        else: correct[a] = 1
                    else:
                        if a in wrong : wrong[a] += 1
                        else: wrong[a] += 1

        if config.dataset_name == 'genome':
            print "accuracy for epoch %d : %.2f" %(epoch, accuracy/tot)
            print "Some Top Correct Answers Are"
            correct = sorted(correct.items(), key=operator.itemgetter(1))
            correct.reverse()
            for i in range(10):
                print correct[i]
            print "Some Top Wrong Answers Are"
            wrong = sorted(wrong.items(), key=operator.itemgetter(1))
            wrong.reverse()
            for i in range(10):
                print wrong[i]

        ops.reset_default_graph()
    	sess.close()

        if config.dataset_name == 'mscoco':
            json.dump(result, open(config.result_path%(epoch), 'wb'))

        print "*** Success to run for Epoch %d ***"%(epoch)

    for epoch in epoch_list:
        t = time.time()
        test_single(epoch)
        print "Time : %dmin"%((time.time()-t)/60)

if __name__ == '__main__':
    #test(config = Config(config_name = 'concat'))
    test(config = Config(config_name = 'mcb'))

