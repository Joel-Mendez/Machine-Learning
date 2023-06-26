from math import log2
import numpy as np
from data2 import *

class Tree:
    def __init__(self,nodeData,value=None,parent=None,depthLimit=None,features=None):
        self.groot = findGroot(self,parent)
        self.data = nodeData
        # print(self.data)
        # print(np.shape(self.data))
        self.nleafs = 0
        self.leafs = []
        self.depthLimit = depthLimit
        self.maxDepth = 0
        self.parent = parent
        self.branch = nodeBranch(value)
        self.depth = nodeDepth(self.parent)
        self.features = features
        self.informationGain = calculateInformationGain(self)
        self.root = findRoot(self.data,self.informationGain,self)
        self.label = nodeLabel(self)

        self.children = childBirth(self)
        updateMaxDepth(self)

    def getEntropy(self):
        return calculateEntropy(self.data)

    def getRoot(self,output):
        if output == 'Name':
            return self.root
        elif output == 'InformationGain':
            return self.informationGain

    def getCommonLabel(self):
        labels = []
        for leaf in self.leafs:
            labels.append(leaf.label)
        uniqueLabels = set(labels)
        CommonLabel = ''
        Occurances = 0
        for label in uniqueLabels:
            if labels.count(label) > Occurances:
                CommonLabel = label
                Occurances = labels.count(label)
        return CommonLabel

def checkAccuracy(tree,data):
    nCorrect = 0;
    rows = np.shape(data.raw_data)[0]
    for i in list(range(0,rows)):
        sample = data.raw_data[i][:]
        actualLabel = sample[0]
        actualLabel = sample[0]
        label = findLeafLabel(sample,tree)
        if label == actualLabel:
            nCorrect = nCorrect + 1
    return nCorrect/rows

def predict_label(X,forest):
    # print('In Predict Label')
    Ypos = 0
    Yneg = 0
    for tree in forest:
        label = findLeafLabel(X,tree)
        # print('Label: ',label)
        if label == 1:
            Ypos = Ypos + 1
        elif label == -1:
            Yneg = Yneg + 1
    if Ypos >= Yneg:
        return 1
    else:
        return -1

def findLeafLabel(sample,node):
    # print(np.shape(sample))
    if node.label != 'none':
        return node.label
    else:
        if sample[node.root] == float(node.children[0].branch):
            return float(node.children[0].label)
        elif sample[node.root] == float(node.children[1].branch):
            return float(node.children[1].label)


def calculateEntropy(data):
    Samples = data[:,0]
    Occurances = np.unique(Samples,return_counts=1)[1]
    entropy = 0
    nSamples = len(Samples)
    for n in Occurances:
        entropy = entropy - (n/nSamples)*log2(n/nSamples)
    return entropy


def calculateInformationGain(self):
    InformationGain = []
    data = self.data
    attributes = self.features
    index = 1;
    # print('attributes',self.features)
    for attribute in attributes:
        labelEntropy = calculateEntropy(data)
        samples = data[:,index]
        # print('data size', np.shape(data))
        # print('index',index)
        # print('Samples,',samples)
        values = [0,1]
        occurances = np.unique(samples,return_counts=1)[1]
        # print('Samples,',samples)
        # print(np.shape(samples))
        # print('Occurances,',occurances)
        # print(np.shape(occurances))
        # print(np.shape(occurances)[0]==1)
        if np.shape(samples)[0]==0:
            print('error')
            continue
            occurances = np.array((0,0))
        if np.shape(occurances)[0]==1:
            if samples[0] == 0:
                # print('samples',samples[0])
                occurances = np.array((occurances[0],0))
                # print(occurances)
            else:
                # print('samples',samples[0])
                occurances = np.array((0,occurances[0]))
                # print(occurances)
        IG = labelEntropy
        for value in values:
            valueData = np.zeros((1,len(attributes)+1))
            for n in list(range(len(samples))):
                if data[n,index] == value:
                    new_data = data[n,:]
                    new_data = np.reshape(new_data,(1,len(attributes)+1))
                    valueData = np.append(valueData,new_data,axis=0)
            valueData = valueData[1:,:]
            valueEntropy = calculateEntropy(valueData)
            p = (occurances[value]/len(samples))
            IG = IG - (p)*valueEntropy
        InformationGain.append(round(float(IG),3))
        index = index + 1
    return InformationGain

def updateMaxDepth(self):
    if self.depth > self.groot.maxDepth:
        self.groot.maxDepth = self.depth

def nodeDepth(parent):
    if parent == None:
        depth = 0
    else:
        depth = parent.depth + 1
    return depth

def  nodeBranch(value):
    if value == None:
        branch = 'root'
    else:
        branch = str(value)
    return branch

def findGroot(self,parent):
    if parent == None:
        return self
    else:
        return parent.groot

def findRoot(data,IG,self):
    max_index = IG.index(max(IG))
    attribute = self.features[max_index]
    return attribute

def findLabel(node):
    samples = node.data[:,0]
    values = np.unique(samples,return_counts=1)[0]
    occurances = np.unique(samples,return_counts=1)[1]
    max_index = int(np.where(occurances==max(occurances))[0][0])
    return values[max_index]

def nodeLabel(self):
    samples = self.data[:,0]
    labelValues = np.unique(samples)
    if (len(labelValues) == 1):
        label = labelValues[0]
        self.groot.nleafs = self.groot.nleafs + 1
        self.groot.leafs.append(self)
    elif(self.depth == self.groot.depthLimit):
        label = findLabel(self)
        self.groot.nleafs = self.groot.nleafs + 1
        self.groot.leafs.append(self)
    else:
        label = 'none'
    return label

def nodeChildren(node):
    if node.label != 'none':
        children = 'none'
    else:
        children = childBirth(node)

def childBirth(parent):
    if parent.label != 'none':
        children ='none'
    else:
        children = [];
        attributes = parent.features
        attribute = parent.root
        # print('parent root',parent.root)
        # print('parent.features',parent.features)
        # print('parent features shape',np.shape(parent.features))
        attribute_index = parent.features.index(attribute)
        # print('attribute index',attribute_index)
        # print('parent data shape',np.shape(parent.data))
        samples = parent.data[:,attribute_index+1]
        # print('parent shape',np.shape(samples))
        values = []
        if 0 in samples:
            values.append(0)
        if 1 in samples:
            values.append(1)
        ## where you stopped debugging...
        for value in values:
            # print('value:',value)
            childData = np.zeros((1,len(attributes)+1))

            for n in list(range(len(samples))):
                # print('n',n)
                # if parent.data[n,attribute_index+1] == value:
                if samples[n] == value:
                    new_data = parent.data[n,:]
                    new_data = np.reshape(new_data,(1,len(attributes)+1))
                    childData = np.append(childData,new_data,axis=0)
            # if np.shape(childData)[0] != 1:
            childData=childData[1:,:]
            child = Tree(childData,value = value,parent =parent,features=parent.features)
            children.append(child)
    return children

# def crossValidation(tree,fold,depths,):
#     #tuple to hold accuracies for each fold
#     # accuracies = ()
#     Accuracy = {}
#     #looping through folds
#     for i in list(range(len(fold))):
#         print('Fold #: ',i+1)
#         foldAccuracy = []
#         foldData = 0
#         excludeIndex = len(fold)-1-i
#         #creating test_fold
#         for j in list(range(len(fold))): # creating test_fold
#             if j != excludeIndex:
#                 if foldData == 0:
#                     foldData = Data(fold[j])
#                 else:
#                     foldData = Data(fold[j],data=None,concatenate=1,oldData = foldData.raw_data)
#         testData = Data(fold[excludeIndex])
#         #looping through depths
#         for depth in depths:
#             print('Depth: ',depth)
#             foldTree = Tree(foldData,None,None,depth)
#             foldAccuracy.append(checkAccuracy(foldTree,testData))
#         Accuracy[i] = foldAccuracy
#     return Accuracy

def outputAccuracies(Accuracies, depths):
    for i in list(range(len(depths))):
        for fold in Accuracies.keys():
            print('Depth of ',depths[i],': Fold',fold+1,' Accuracy - ',Accuracies[fold][i])
        print('')

def findBestDepth(Accuracies,depths):
    bestDepth = 0
    bestAccuracy = 0
    for i in list(range(len(depths))):
        depthAccuracy = 0
        sumAccuracy = 0
        n = 0
        for fold in Accuracies.keys():
            sumAccuracy = sumAccuracy + Accuracies[fold][i]
            n = n+1
        depthAvgAccuracy = sumAccuracy/n
        if depthAvgAccuracy > bestAccuracy:
            bestDepth = depths[i]
            bestAccuracy = depthAvgAccuracy
    return [bestDepth, bestAccuracy]
