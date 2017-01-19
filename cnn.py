import sys
sys.path.append("/home/whyjay/caffe/python")
sys.path.append("/usr/lib/python2.7/dist-packages/")
import caffe
import numpy as np
#import skimage
#import cv2
from utils import *
from tqdm import tqdm

deploy = '/home/shmsw25/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
model = '/home/shmsw25/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
mean = '/home/shmsw25/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

class CNN(object):
    def __init__(self, deploy=deploy, model=model, mean=mean,
                 batch_size=10, width=227, height=227):
        self.deploy = deploy
        self.model = model
        self.mean = mean

        self.batch_size = batch_size
        self.net, self.transformer = self.get_net()
        self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

        self.width = width
        self.height = height

    def get_net(self):
        caffe.set_mode_gpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)

        transformer = caffe.io.Transformer({
            'data':net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

        return net, transformer

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        iter_until = len(image_list) + self.batch_size
        all_feats = np.zeros([len(image_list)] + layer_sizes)

        starts = range(0, iter_until, self.batch_size)[:-1]
        ends = range(self.batch_size, iter_until, self.batch_size)

        for i in tqdm(range(len(starts))):
            start = starts[i]
            end = ends[i]

            image_batch_file = image_list[start:end]
            image_batch = np.array(map(
                lambda x: crop_image(
                    x, target_width=self.width, target_height=self.height),
                image_batch_file))

            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]],
                                dtype=np.float32)

            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.transformer.preprocess('data', in_)

            out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
	    reshape_size = [-1]+layer_sizes
	    all_feats[start:end] = out[layers].reshape(reshape_size)

        return all_feats
