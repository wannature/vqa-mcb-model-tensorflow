# vqa-mcb-model-tensorflow

This is an implementation of [Frame Augmented Alternating Attention Network for Video Question Answering] with Tensorflow. 

### Codes

Before Training

- preprocess_msrvttqa.py : build MSCOCO annotations for train(with idx of image feature vector, question, and idx of answer) and feature vectors. Also build word2ix and ix2word for questions and answers.
- preprocess_msvdqa.py : build MSCOCO annotations for valid(with image_id, idx of image feature vector, question, question_id) and feature vectors.
- build_genome_datasets.py : build Visual Genome annotations for train & valid.

Models and Train & Test codes(Tensorflow is used)

- config.py : configuration file 
- faster_rcnn: extract the spatial regions for question to region attention.
- model
  - faa.py : a MCB model which do not have attention mapping
  - run_faa.py : a code for training and testing.




