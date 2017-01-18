import sys, os
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import skimage.io as io
import json, random
from config import Config

# set up file names and paths
config = Config()
annFile = config.val_annotations_path
quesFile = config.val_questions_path
resFile = config.result_path

vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate()

# print accuracies
print "\n"
print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"

