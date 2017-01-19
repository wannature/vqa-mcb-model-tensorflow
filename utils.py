import sys
sys.path.append("/home/whyjay/caffe/python")
sys.path.append("/usr/lib/python2.7/dist-packages/")
import cv2
import numpy as np
import skimage
from IPython import embed

def crop_image(x, target_height=227, target_width=227):
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))

    elif height < width:
        resized_image = cv2.resize(
            image,
            (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(
            image,
            (target_height, int(height * float(target_width)/width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))


def build_word_vocab(sentence_iterator, word_count_threshold=10): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def shuffle_block_df(df):
    index = list(df.index)
    block_idx = range(len(index)/5)
    np.random.shuffle(block_idx)

    new_index = []
    for b in block_idx:
        new_index += index[5*b : 5*(b+1)]
    df = df.ix[new_index]
    return df.reset_index(drop=True)

def shuffle_df(df):
    index = list(df.index)
    np.random.shuffle(index)
    df = df.ix[index]
    return df.reset_index(drop=True)

def prep_cocoeval_flickr(ann_df, res_df):
    pass
# ann_df uniq image images
#ann = {'images':None, 'info':'', 'type':'captions', 'annotations':None}
#ann_caps = {'caption':"afjiwel", 'id':1, 'image_id':2}
#ann_images = {'id':2}
#res = [{'caption':'hello', 'image_id':2}, {}]
# return metric
