import json, operator, re, time
import tensorflow as tf, numpy as np
from config import Config

def test(config=Config(), epoch_list = range(10)):

    from_idx = range(0, config.valid_num, config.val_batch_size)
    to_idx = range(config.batch_size, config.valid_num, config.batch_size)

    annotations_result = json.load(open(config.val_annotations_result_path, 'rb'))
    image_idx = annotations_result['image_idx']
    image_ids = annotations_result['image_ids']
    questions = annotations_result['questions']
    question_ids = annotations_result['question_ids']
    q_word2ix = pickle.load(open(config.worddic_path+'q_word2ix', 'rb'))
    a_ix2word = pickle.load(open(config.worddic_path+'a_ix2word', 'rb'))
    imgix2featix = pickle.load(open(config.val_imgix2featix, 'rb'))
    feats = np.load(config.val_feats_path)

    def test_single(epoch):
        print "*** Test Start for Epoch %d ***" %(epoch)
        sess = Session()
        model = config.vqamodel(
            batch_size=config.val_batch_size,
            feature_dim=config.feature_dim,
            proj_dim=config.proj_dim,
            word_num=config.word_num,
            embed_dim=config.embed_dim,
            ans_candi_num=config.ans_candi_num,
            n_lstm_steps=config.n_lstm_steps)

        image_feat, question, max_prob_words = model.solver()
        saver = tf.train.Saver(max_to_keep = 50)
        saver.restore(sess, config.model_path%(epoch))

        result = []
        for (start, end) in zip(from_idx, to_idx):
            # make curr_image_feat [batch_size, feature_dim]
            curr_image_feat = feats[imgix2featix[image_id[start:end]]]
            # make curr_question [batch_size, n_lstm_steps]
            curr_question = map(lambda ques :
                [q_word2ix[word] for word in ques.lower() if word in q_word2ix],
                questions[start:end])
            curr_question = sequence.pad_sequences(
                curr_question, padding='post', maxlen=config.n_lstm_steps)

            answer_ids = sess.run(max_prob_words,
                feed_dict = {image_feat : curr_image_feat,
                            question : curr_question})

            answers = map(lambda ix : a_ix2word[ix], answer_ids)
            for i in range(config.batch_size):
                result.append({
                    'question_id' : question_ids[start+i],
                    'answer' : answers[i]
                    })

        pickle.dump(result, config.result_path%(epoch))
        print "*** Success to run for Epoch %d ***"%(epoch)

    for epoch in epoch_list:
        t = time.time()
        test_single(epoch)
        print "Time : %dmin"%((time.time()-t)/60)

if __name__ == '__main__':
    test()


