import numpy as np
import random

def sgd(data,error_threshold,Wo,learning_rate):
    nSamples = np.shape(data)[0]
    nFeatures = np.shape(data)[1]-1
    W = np.full((1,nFeatures),Wo)
    W = np.array([.001,.001,-.01,-.01,.1])
    error = 999
    epoch = 1
    index_list = list(range(0,nSamples))

    while(error>error_threshold and epoch <= 100):
        random.shuffle(index_list)
        # print('In Epoch #',epoch)
        # for i in list(range(1,nSamples)):
        error = 0
        for i in index_list:
            Xi = data[i-1,1:] #Sample i
            Xi = Xi.astype(float)
            Yi = float(data[i,0]) #Label i
            Yp = np.sum(W*Xi) #Predicted Label
            # error = (Yi-Yp) #Error
            W = W + learning_rate*(Yi-Yp)*Xi #Updating weight vector
            # print('/////////////////////////////////////////////////////////////////////////////////////////')
            # print('Xi: ',Xi)
            # print('Yi: ',Yi)
            # print('Yp: ',Yp)
            # print('W: ', W)
            # print('Iteration: ',i,', Error: ', error)
            #error = abs(error)
            error = error + abs(Yi-Yp)

        error = error/nSamples
        # print('Epoch #',epoch,'Error: ',error)
        epoch = epoch + 1

    # print('Broke out of the looooop!')
    # print('W: ', W)
    return W

def crossValidation(folds,LEARNING_RATE=None):
    Error_Dict={}
    for lr in LEARNING_RATE:
        print('/////////////////////////////////////////////////////////////')
        print('Learning Rate = ', lr)
        Error_Dict[(lr)] = []
        for nFold in list(range(len(folds))):
            # print('Learning Rate: ',lr,'Fold #',nFold+1)
            excludeIndex = nFold
            trainFold, testFold = createFold(folds,excludeIndex)
            W = sgd(trainFold,error_threshold=.01, Wo = 0.001, learning_rate = lr)
            error = getError(testFold,W)
            Error_Dict[(lr)].append(error)
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
    print('Errot Dictionary: ',Error_Dict)
    bestHyperParameters = 0
    minError = 99999
    print('Dictionary keys, ', Error_Dict.keys())
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
