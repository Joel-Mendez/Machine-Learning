from math import log2
import numpy as np
from data import *

class Tree:
    def __init__(self,nodeData,value=None,parent=None,depthLimit=None, GI = 0):
        self.groot = findGroot(self,parent)
        self.data = nodeData
        self.nleafs = 0
        self.leafs = []
        self.depthLimit = depthLimit
        self.maxDepth = 0
        self.parent = parent
        self.branch = nodeBranch(value)
        self.depth = nodeDepth(self.parent)
        if GI ==1:
            self.informationGain = calculateInformationGainGI(self)
        else:
            self.informationGain = calculateInformationGain(self)
        self.root = findRoot(self.data,self.informationGain)
        self.label = nodeLabel(self)
        self.children = childBirth(self)
        updateMaxDepth(self)

    def getEntropy(self):
        return calculateEntropy(self.data)

    def getGini(self):
        return calculateGiniIndex(self.data)

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

def findLeafLabel(sample,node):
    if node.label != 'none':
        return node.label
    else:
        i = node.data.column_index_dict[node.root]
        for child in node.children:
            if sample[i] == child.branch:
                return findLeafLabel(sample,child)

def calculateEntropy(data):
    Samples = data.get_column('label')
    Occurances = np.unique(Samples,return_counts=1)[1]
    entropy = 0
    nSamples = len(Samples)
    for n in Occurances:
        entropy = entropy - (n/nSamples)*log2(n/nSamples)
    return entropy

def calculateGiniIndex(data):
    Samples = data.get_column('label')
    Occurances = np.unique(Samples,return_counts=1)[1]
    GI = 1
    nSamples = len(Samples)
    for n in Occurances:
        GI = GI - (n/nSamples)**2
    return GI

def calculateInformationGain(self):
    InformationGain = []
    data = self.data
    attributes = data.column_index_dict
    for attribute in attributes.keys():
        if attribute != 'label':
            labelEntropy = calculateEntropy(data)
            samples = data.get_column(str(attribute))
            values = np.unique(samples,return_counts=1)[0]
            occurances = np.unique(samples,return_counts=1)[1]
            IG = labelEntropy
            for value in values:
                i = int(np.where(values==value)[0])
                valueData = data.get_row_subset(attribute,value)
                valueEntropy = calculateEntropy(valueData)
                p = (occurances[i]/len(samples))
                IG = IG - (p)*valueEntropy
            InformationGain.append(round(float(IG),3))
    return InformationGain

def calculateInformationGainGI(self):
    InformationGain = []
    data = self.data
    attributes = data.column_index_dict
    for attribute in attributes.keys():
        if attribute != 'label':
            labelEntropy = calculateGiniIndex(data)
            samples = data.get_column(str(attribute))
            values = np.unique(samples,return_counts=1)[0]
            occurances = np.unique(samples,return_counts=1)[1]
            IG = labelEntropy
            for value in values:
                i = int(np.where(values==value)[0])
                valueData = data.get_row_subset(attribute,value)
                valueEntropy = calculateGiniIndex(valueData)
                p = (occurances[i]/len(samples))
                IG = IG - (p)*valueEntropy
            InformationGain.append(round(float(IG),3))
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

def findRoot(data,IG):
    max_index = IG.index(max(IG))+1
    attribute = data.index_column_dict[max_index]
    return attribute

def findLabel(node):
    samples = node.data.get_column('label')
    values = np.unique(samples,return_counts=1)[0]
    occurances = np.unique(samples,return_counts=1)[1]
    max_index = int(np.where(occurances==max(occurances))[0][0])
    return values[max_index]

def nodeLabel(self):
    samples = self.data.get_column('label')
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
        attribute = parent.root
        samples = parent.data.get_column(str(attribute))
        values = np.unique(samples)
        for value in values:
            childData = parent.data.get_row_subset(attribute,value)
            child = Tree(childData,value,parent)
            children.append(child)
    return children

def crossValidation(tree,fold,depths,):
    #tuple to hold accuracies for each fold
    # accuracies = ()
    Accuracy = {}
    #looping through folds
    for i in list(range(len(fold))):
        print('Fold #: ',i+1)
        foldAccuracy = []
        foldData = 0
        excludeIndex = len(fold)-1-i
        #creating test_fold
        for j in list(range(len(fold))): # creating test_fold
            if j != excludeIndex:
                if foldData == 0:
                    foldData = Data(fold[j])
                else:
                    foldData = Data(fold[j],data=None,concatenate=1,oldData = foldData.raw_data)
        testData = Data(fold[excludeIndex])
        #looping through depths
        for depth in depths:
            print('Depth: ',depth)
            foldTree = Tree(foldData,None,None,depth)
            foldAccuracy.append(checkAccuracy(foldTree,testData))
        Accuracy[i] = foldAccuracy
    return Accuracy

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
