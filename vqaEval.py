import sys, os
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import json, random
from config import Config

def vqaEval(config = Config(), epoch_list = range(3,10)):
    accuracy_dic = {}
    best_accuracy, best_epoch = 0.0, -1

    # set up file names and paths
    annFile = config.selected_val_annotations_path
    quesFile = config.selected_val_questions_path

    for epoch in epoch_list:

        resFile = config.result_path%(epoch)

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
        accuracy = vqaEval.accuracy['overall']
        print "Overall Accuracy is: %.02f\n" %(accuracy)
        print "Per Question Type Accuracy is the following:"
        for quesType in vqaEval.accuracy['perQuestionType']:
    	    print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
        print "\n"
        print "Per Answer Type Accuracy is the following:"
        for ansType in vqaEval.accuracy['perAnswerType']:
	    print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
        print "\n"

        accuracy_dic[epoch] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch


    print "** Done for every epoch! **"
    print "Accuracy Dictionry"
    print accuracy_dic
    print "Best Epoch is %d with Accuracy %.02f"%(best_epoch, best_accuracy)


if __name__ == '__main__':
    vqaEval()

