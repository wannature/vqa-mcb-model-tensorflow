# vqa-mcb-model-tensorflow

This is an implementation of [Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Groudning](https://arxiv.org/abs/1606.01847) with Tensorflow. A theorical background of this paper is in [this paper](https://github.com/mikeOxO/vqa-mcb-model-tensorflow/blob/master/mcb_for_vqa.pdf).  

### Codes

Before Training(Caffe is used)

- build_train_datasets.py : build MSCOCO annotations for train(with idx of image feature vector, question, and idx of answer) and feature vectors. Also build word2ix and ix2word for questions and answers.
- build_val_datasets.py : build MSCOCO annotations for valid(with image_id, idx of image feature vector, question, question_id) and feature vectors.
- build_genome_datasets.py : build Visual Genome annotations for train & valid.

Models and Train & Test codes(Tensorflow is used)

- config.py : configuration file (support model without Attention & with Attention, support dataset MSCOCO & Visual Genome)
- model
  - vqamodel.py : an abstract model for VQA
  - mcb_vqamodel.py : a MCB model which do not have attention mapping
  - mcbAtt_vqamodel.py : a MCB model which have attention mapping
  - concat_vqamodel.py : a non-bilinear model(Use concatenate to make feature)
  - CBP : Module for compact bilinear pooling from [here](github.com/therne/compact-bilinear-pooling-tf).
- train.py : a code for training
- test.py : a code for test
- vqaEvaluation, vqaTools : Metric Evaluation Tool for VQA dataset from [here](https://github.com/VT-vision-lab/VQA/).

### MSCOCO Datasets

MSCOCO Datasets are available on [here](http://visualqa.org/download.html). Only Real Images and OpenEnded questions are used.
3 questions per an image, 10 answers per a question.
**I only used 50000 images and 150000 questions. Split training & validation set with 9 : 1 rate.**

- In questions, 10525 different words exist and most frequent 5000 words are selected.(threshold 3) In paper, 13K~20K words were selected.
- In answers, 50697 different words exist and most frequent 5000 words are selected.(threshold 9) In paper, 3000 words were selected.
- Length of questions
  - 0~9 : 126520, 10~19 : 8465, 20~29 : 16
  - Consider only 20 words in front when embedding questions
- Use answers with confidence 'yes' only : Total 1049879 numbers of (Image, Question, Answer) are used.

### VISUAL GENOME Datasets

Visual Genome Datasets are available on [here](http://visualgenome.org/api/v0/api_home.html). The answers have multiple length but only answers with one word are used.
Many questions per an image, 1 answer per a question.
**I used 68990 images and 611209 questions for training and 9417 images and 35032 questions for validation.**

- In questions, 17112 different words exist and most frequent 8000 words are selected.(threshold 3).
- In answers, 32686 different words exit and most frequent 5000 words are selected.(threshold 3)
- Length of questions
  - 0~9 : 1185284, 10~19 : 42965, 20~29 : 37
