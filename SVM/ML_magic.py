import numpy as np
import random
from decisiontree import *
from math import log
# SVM Over Trees
# def svm():

def accuracy(X, Y, W):
    nSamples = np.shape(X)[0]
    nCorrect = 0
    for i in list(range(0,nSamples)):
        Yp = np.sum(W*X[i,:])
        if np.sign(Yp) == np.sign(Y[i]):
            nCorrect = nCorrect + 1
    acc = nCorrect/nSamples
    return acc

def accuracy_forest(X, Y, forest):
    nSamples = np.shape(X)[0]
    nCorrect = 0
    for i in list(range(0,nSamples)):
        Yp = predict_label(X[i,:],forest)
        if np.sign(float(Yp)) == np.sign(float(Y[i])):
            nCorrect = nCorrect + 1
    acc = nCorrect/nSamples
    # print('Accuracy:',acc)
    return acc


def accuracy_svm(X, Y, W, forest):
    nSamples = np.shape(X)[0]
    nTrees = len(forest)
    P = np.zeros((nSamples,nTrees))
    for i in list(range(nSamples)):
        for j in list(range(nTrees)):
            P[i,j] = findLeafLabel(X[i,:],forest[j])
    nCorrect = 0
    for i in list(range(0,nSamples)):
        Yp = np.sum(W*P[i,:])
        if np.sign(Yp) == np.sign(Y[i]):
            nCorrect = nCorrect + 1
    acc = nCorrect/nSamples
    return acc


def addBias(dataset):
    bias = np.ones((np.shape(dataset)[0],1))
    dataset = np.concatenate([bias,dataset],1)
    return dataset

def createFold(x_fold,y_fold, excludeIndex):
    train_X_empty = 1
    test_X = x_fold[excludeIndex]
    test_Y = y_fold[excludeIndex]
    for i in list(range(len(x_fold))):
        if i != excludeIndex:
            if train_X_empty == 1:
                train_X = x_fold[i]
                train_Y = y_fold[i]
                train_X_empty = 0
            else:
                train_X = np.concatenate((train_X,x_fold[i]),axis=0)
                train_Y = np.concatenate((train_Y,y_fold[i]),axis=0)
    return train_X, train_Y, test_X, test_Y

def crossValidation(name,x_fold,y_fold,HP1=None,HP2=None,forest=None):
    Accuracy_Dict={}
    if name == "sgd":
        for hp1 in HP1:
            for hp2 in HP2:
                print('Cross-Validation: Learning Rate = ', hp1, ', Loss Tradeoff = ',hp2)
                Accuracy_Dict[(hp1,hp2)] = []
                for nFold in list(range(len(y_fold))):
                    excludeIndex = len(y_fold)-1-nFold
                    train_X, train_Y, test_X, test_Y = createFold(x_fold,y_fold,excludeIndex)
                    W = sgd(train_X,train_Y,hp1,hp2)
                    acc = accuracy(test_X,test_Y,W)
                    Accuracy_Dict[(hp1,hp2)].append(acc)
    elif name == "naive_bayes":
        for hp1 in HP1:
            print('Cross-Validation: Smoothing Term = ', hp1)
            Accuracy_Dict[(hp1)] = []
            for nFold in list(range(len(y_fold))):
                # print('Fold #',nFold+1)
                excludeIndex = len(y_fold)-1-nFold
                train_X, train_Y, test_X, test_Y = createFold(x_fold,y_fold,excludeIndex)
                W = naive_bayes(train_X,train_Y,hp1)
                acc = accuracy(test_X,test_Y,W)
                Accuracy_Dict[(hp1)].append(acc)
    elif name == "random_forest":
        for hp1 in HP1:
            print('Forest Size = ', hp1)
            Accuracy_Dict[(hp1)] = []
            for nFold in list(range(len(y_fold))):
                # print('Fold #',nFold+1)
                excludeIndex = len(y_fold)-1-nFold
                train_X, train_Y, test_X, test_Y = createFold(x_fold,y_fold,excludeIndex)
                forest = random_forest(train_X,train_Y,SampleSize=100,FeatureSize=50,k=hp1)
                acc = accuracy_forest(test_X,test_Y,forest)
                Accuracy_Dict[(hp1)].append(acc)
    elif name == "svm":
        for hp1 in HP1:
            for hp2 in HP2:
                print('Learning Rate = ', hp1, ', Loss Tradeoff = ',hp2)
                Accuracy_Dict[(hp1,hp2)] = []
                for nFold in list(range(len(y_fold))):
                    # print('Fold #',nFold+1)
                    excludeIndex = len(y_fold)-1-nFold
                    train_X, train_Y, test_X, test_Y = createFold(x_fold,y_fold,excludeIndex)
                    W = svm(train_X,train_Y,forest,hp1,hp2)
                    acc = accuracy_svm(test_X,test_Y,W,forest)
                    Accuracy_Dict[(hp1,hp2)].append(acc)
    return Accuracy_Dict

def findBestHyperParameters(Accuracy_Dict):
    bestHyperParameters = 0
    bestAccuracy = 0
    for combination in Accuracy_Dict.keys():
        sumAccuracy = 0
        n = 0
        for acc in Accuracy_Dict[combination]:
            sumAccuracy = sumAccuracy + acc
            n = n+1
        accuracy = sumAccuracy/n
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestHyperParameters = combination
    return bestHyperParameters, bestAccuracy

def sgd(X,Y,lr,c):
    nSamples = np.shape(X)[0]
    nFeatures = np.shape(X)[1]
    lr0 = lr
    W = np.full((1,nFeatures),0)
    Wo = W
    epoch = 1
    index_list = list(range(0,nSamples))
    acc = 0
    while(epoch < 25):
        random.shuffle(index_list)
        acc = 0
        for i in index_list:
            Xi = X[i,:]
            Xi = Xi.astype(float)
            Yi = float(Y[i])
            Yp = np.sum(W*Xi)
            if Yi*Yp<=1:
                W = (1-lr)*W+lr*c*Yi*Xi
            elif Yi*Yp>1:
                acc = acc +1
                W = (1-lr)*W
        lr = lr0 / (1+epoch)
        acc = acc/nSamples
        epoch = epoch + 1
    return W

def binarize(X):
    for i in list(range(np.shape(X)[0])):
        for j in list(range(np.shape(X)[1])):
            if j == 0:
                X[i,j] = X[i,j]
            elif X[i,j] < 500:
                X[i,j] = 0
            elif X[i,j] >= 500:
                X[i,j] = 1
    return X


def naive_bayes(X,Y,s):
    nSamples = np.shape(X)[0]
    nFeatures = np.shape(X)[1]
    Ypos = 0
    Yneg = 0
    Xpos = np.full((1,nFeatures),0)
    Xneg = np.full((1,nFeatures),0)
    for i in list(range(0,nSamples)):
        if Y[i] == 1:
            Ypos = Ypos + 1
            Xpos = Xpos + X[i,:]
        else:
            Yneg = Yneg + 1
            Xneg = Xneg + X[i,:]
    Ppos = (Xpos+s)/(Ypos+2*s)
    Pneg = (Xneg+s)/(Yneg+2*s)
    PYpos = Ypos / (Ypos+Yneg)
    PYneg = 1 - PYpos
    W = np.full((1,nFeatures),0)
    for i in list(range(nFeatures)):
        # W[0,i] = log( (Ppos[0,i])/ (Pneg[0,i])) *(PYpos/PYneg) #.8 and 0
        W[0,i] = log( (Ppos[0,i])/ (Pneg[0,i])) #.78 and 0
        # W[0,i] = log( (Ppos[0,i]*PYpos)/ (Pneg[0,i]*PYneg)) #.66 and 0
        # W[0,i] = np.sign(log( (Ppos[0,i])/ (Pneg[0,i])) *(PYpos/PYneg)) #.74 and .5
    return W

def random_forest(X,Y,SampleSize,FeatureSize,k):
    Forest = []
    #X = Samples; Y = Labels; k = # of trees
    nSamples = np.shape(X)[0]
    nFeatures = np.shape(X)[1]
    for n in list(range(0,k)):
        # Randomly Selecting Samples
        sampleIndex = random.sample(range(nSamples),SampleSize)
        samples = np.zeros((SampleSize,nFeatures))
        training_labels = np.zeros((SampleSize,1))

        for i in list(range(0,SampleSize)):
            samples[i,:] = X[sampleIndex[i],:]
            training_labels[i,:]=Y[sampleIndex[i]]

        # Randomly Selecting Features
        featureIndex = random.sample(range(nFeatures),FeatureSize)
        training_samples = np.zeros((SampleSize,FeatureSize))
        for i in list(range(0,FeatureSize)):
            training_samples[:,i]=samples[:,featureIndex[i]]
        training_data = np.concatenate((training_labels,training_samples),axis=1)
        #Creating Tree
        tree = Tree(training_data,value=None, parent=None, depthLimit=1,features=featureIndex)
        # print(tree)
        Forest.append(tree)
    return Forest

def svm(X,Y,forest,lr,c):
    nSamples = np.shape(X)[0]
    nTrees = len(forest)
    P = np.zeros((nSamples,nTrees))
    for i in list(range(nSamples)):
        for j in list(range(nTrees)):
            P[i,j] = findLeafLabel(X[i,:],forest[j])
    W = sgd(P,Y,lr,c)
    return W
