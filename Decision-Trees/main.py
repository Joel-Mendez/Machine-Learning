from data import *
from decisiontree import *
import numpy as np

training_data = Data("train.csv")
test_data = Data("test.csv")
foldPaths = ("fold1.csv","fold2.csv","fold3.csv","fold4.csv","fold5.csv")
depths = [1,2,3,4,5,10,15]
Groot = Tree(training_data,value = None, parent=None, depthLimit = None)

#ENTROPY METHOD #########################################################
print('ENTROPY METHOD::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
print('(a) Most common label of the data-------------------------')
print('Common Label: ', Groot.getCommonLabel())
print('')

print('(b) Entropy of the data-----------------------------------')
print('Entropy: ', round(Groot.getEntropy(),3))
print('')

print('(c) Best feature and its information gain-----------------')
print('Best feature: ', Groot.getRoot('Name'))
print('Information Gain: ', Groot.getRoot('InformationGain'))
print('')

print('(d) Accuracy on the training set--------------------------')
print(checkAccuracy(Groot,training_data))
print('')

print('(e) Accuracy on the test set------------------------------')
print('Accuracy: ', checkAccuracy(Groot,test_data))
print('')

#GINI METHOD #########################################################
GrootGI = Tree(training_data,value = None, parent=None, depthLimit = None,GI=1)
print('GINI INDEX METHOD::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
print('GiniIndex: (a) Most common label of the data-------------------------')
print('Common Label: ', GrootGI.getCommonLabel())
print('')

print('GiniIndex: (b) GI of the data-----------------------------------')
print('Entropy: ', round(GrootGI.getGini(),3))
print('')

print('GiniIndex: (c) Best feature and its information gain-----------------')
print('Best feature: ', GrootGI.getRoot('Name'))
print('Information Gain: ', GrootGI.getRoot('InformationGain'))
print('')

print('GiniIndex: (d) Accuracy on the training set--------------------------')
print(checkAccuracy(GrootGI,training_data))
print('')

print('GiniIndex: (e) Accuracy on the test set------------------------------')
print('Accuracy: ', checkAccuracy(GrootGI,test_data))
print('')

#CROSS VALIDATION #####################################################
print('CROSS VALIDATION::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
print('(f) Cross-validation accuracies for each fold-------------')
accuracies = crossValidation(Groot,foldPaths,depths)
outputAccuracies(accuracies,depths)
print('')

print('(g) Best Depth--------------------------------------------')
bestDepth = findBestDepth(accuracies,depths)
print('Best Tree Depth: ', bestDepth[0])
print('Average Accuracy: ', bestDepth[1])
print('')

print('(h) Accuracy on test set using the best depth ------------')
Groot = Tree(training_data,value = None, parent=None, depthLimit = bestDepth)
print('Accuracy: ',checkAccuracy(Groot,test_data))
