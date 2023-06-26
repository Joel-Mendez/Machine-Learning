import numpy as np
import random
from math import sqrt


def sgd(data,error_threshold,lr,c):
    nSamples = np.shape(data)[0]
    nFeatures = np.shape(data)[1]-1
    lr0 = lr
    # W = np.array([.001,.001,.001, .001, .001, .001,.001,.001,.001])
    W = np.full((1,nFeatures),0)
    error = 999
    Y = data[:,0]
    X = data[:,1:]
    epoch = 1
    index_list = list(range(0,nSamples))

    EpochList =[]
    WeightList = []
    while(error>error_threshold and epoch <= 10):
        random.shuffle(index_list)
        error = 0
        print('Epoch: ',epoch)
        for i in index_list:
            Xi = X[i,:]
            Xi = Xi.astype(float)
            Yi = float(Y[i])
            Yp = np.sum(W*Xi)
            if Yp > 1:
                Yp = 1
            elif Yp < 0:
                Yp =0
            if abs(Yi-Yp) < error_threshold:
                W = W + lr*(Yi-Yp)*Xi ##+ lr*c*W
            else:
                W = W + lr*(Yi-Yp)*Xi + lr*c*W
            error = error + abs(Yi-Yp)
        error = error/nSamples
        epoch = epoch + 1
        EpochList.append(epoch-1)
        WeightList.append(W)
    return W

def crossValidation(folds,LEARNING_RATE=None,C=None):
    Error_Dict={}
    for lr in LEARNING_RATE:
        for c in C:
            print('/////////////////////////////////////////////////////////////')
            print('Learning Rate = ', lr)
            print('Tradeoff = ',c)
            Error_Dict[(lr,c)] = []
            for nFold in list(range(len(folds))):
                # print('Learning Rate: ',lr,'Fold #',nFold+1)
                print('Fold: ',nFold+1)
                excludeIndex = nFold
                trainFold, testFold = createFold(folds,excludeIndex)
                W = sgd(trainFold,error_threshold=.05,lr=lr,c=c)
                error = getError(testFold,W)
                Error_Dict[(lr,c)].append(error)
    HyperParameters = findBestHyperParameters(Error_Dict)
    print('Best Hyperparameters: ', HyperParameters)
    return HyperParameters


def createFold(folds, excludeIndex):
    train_X_empty = 1
    testFold = folds[excludeIndex].raw_data


    for i in list(range(len(folds))):
        if i != excludeIndex:
            if train_X_empty == 1:
                trainFold = folds[i].raw_data
                train_X_empty = 0
            else:
                trainFold = np.concatenate((trainFold,folds[i].raw_data),axis=0)
    return trainFold, testFold

def getError(data, W):
        nSamples = np.shape(data)[0]
        error = 0
        for i in list(range(1,nSamples)):
            Xi = data[i-1,1:] #Sample i
            Xi = Xi.astype(float)
            Yi = float(data[i,0]) #Label i
            Yp = np.sum(W*Xi) #Predicted Label
            error = error + abs((Yi-Yp)) #Error
        error = error/nSamples
        return error

def findBestHyperParameters(Error_Dict):
    # print('Error Dictionary: ',Error_Dict)
    bestHyperParameters = 0
    minError = 99999
    # print('Dictionary keys, ', Error_Dict.keys())
    for lr in Error_Dict.keys():
        sumError = 0
        n = 0
        for e in Error_Dict[lr]:
            print('LR: ',lr, 'e: ',e)
            sumError = sumError + e
            n = n+1
        error = sumError/n
        if error < minError:
            minError = error
            bestHyperParameters = lr
    return bestHyperParameters
