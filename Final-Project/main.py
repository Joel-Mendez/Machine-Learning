from data import *
import numpy as np
from gradient_search import *
import matplotlib.pyplot as plt
##########################################################################
# Importing Data #########################################################
##########################################################################
train = Data('train_v5.csv')
fold1 = Data('fold1_v5.csv')
fold2 = Data('fold2_v5.csv')
fold3 = Data('fold3_v5.csv')
fold4 = Data('fold4_v5.csv')
fold5 = Data('fold5_v5.csv')
test = Data ('test_v5.csv')

folds = [fold1, fold2, fold3, fold4, fold5]

##########################################################################
# Cross Validation  ######################################################
##########################################################################
LEARNING_RATE = [.00001,.0001,.001]
C = [.0001, .001, .01]
# LEARNING_RATE = [.00001]
print('Hyper-Parameters: Learning Rate = ', LEARNING_RATE)
print('Hyper-Parameters: Loss Trade-off = ', C)
bestHP = crossValidation(folds,LEARNING_RATE=LEARNING_RATE, C=C)
lr = bestHP[0]
C = bestHP[1]
print('Best Learning Rate: ',lr)
print('Best Trade-off Coefficient: ',C)

##########################################################################
# Training  ##############################################################
##########################################################################
print('Training')
W = sgd(train.raw_data,error_threshold=.05,lr = lr,c=C)
# W = Results[0]
# EpochList = Results[1]
# WeightList = Results[2]

# trainingErrorList = []
# testingErrorList = []
# for i in range(len(EpochList)):
#     trainingErrorList.append(getError(train.raw_data,WeightList[i]))
#     testingErrorList.append(getError(test.raw_data,WeightList[i]))


# p1=plt.plot(EpochList,trainingErrorList)
# p2=plt.plot(EpochList,testingErrorList)
# plt.show()


trainingError = getError(train.raw_data,W)
print('Weight: ',W)
print('Training Error: ',trainingError)

##########################################################################
# Testing  ###############################################################
##########################################################################
testingError = getError(test.raw_data,W)
print('Testing Error: ',testingError)
