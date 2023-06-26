import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from IPython.display import HTML
from data import *

# Color settings for plotting
colors = ['darkred', 'royalblue']
colors_region = ['mistyrose', 'lightsteelblue']
cmap = ListedColormap(colors)
cmap_region = ListedColormap(colors_region)


class History:
    def __init__(self, num_epochs):
        self.training_hist = dict()
        self.num_epochs = num_epochs
        self.updates = 0;
        for n in range(num_epochs):
            self.training_hist[n] = {'w_hist': [],
                                'b_hist': [],
                                'acc_hist': [],
                                'point_hist':[]}
    def store(self, x, y, w, b, accuracy, epoch):
        self.training_hist[epoch]['point_hist'].append((x, y))
        self.training_hist[epoch]['w_hist'].append(w.copy())
        self.training_hist[epoch]['b_hist'].append(b)
        self.training_hist[epoch]['acc_hist'].append(accuracy)
        self.updates = self.updates + 1

    def plotLearningCurve(self,save=None):
        epochs = list(range(self.num_epochs))
        epochs = []
        accuracies = [];
        for key in self.training_hist.keys():
            epochs.append(key+1)
            accuracies.append(self.training_hist[key]['acc_hist'][-1])
        epochs = np.asarray(epochs)
        accuracies = np.asarray(accuracies)
        plt.scatter(epochs, accuracies, c=np.ones(epochs.shape[0]), edgecolor='black', cmap=cmap)
        if save != None:
            plt.savefig(save)
        plt.show()


    def getBestClassifier(self):
        Epoch = 0;
        Accuracy = 0;
        Weight = 0;
        Bias = 0;

        for key in self.training_hist.keys():
            if self.training_hist[key]['acc_hist'][-1] > Accuracy:
                Epoch = key+1
                Accuracy = self.training_hist[key]['acc_hist'][-1]
                Weight = self.training_hist[key]['w_hist'][-1]
                Bias = self.training_hist[key]['b_hist'][-1]
        return (Epoch, Accuracy, Weight, Bias)

def accuracy(X, y, w, b):
    yp = predict(X,w,b)
    acc = sum(abs(yp+(y)))/(2*len(y))
    return acc

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

def crossValidation(x_fold,y_fold,epochs=10,LEARNING_RATE=None,MARGIN=None,decay=None,avg=None,pocket=None):
    Accuracy_Dict={}
    if MARGIN == None:
        for lr in LEARNING_RATE:
            print('Learning Rate = ', lr)
            Accuracy_Dict[(lr)] = []
            for nFold in list(range(len(y_fold))):
                print('Fold #',nFold+1)
                excludeIndex = len(y_fold)-1-nFold
                train_X, train_Y, test_X, test_Y = createFold(x_fold,y_fold,excludeIndex)
                w, b, hist = train(train_X,train_Y,epochs=epochs,lr=lr,decay=decay,avg=avg,pocket=pocket,margin=MARGIN)
                acc = accuracy(test_X,test_Y,w,b)
                Accuracy_Dict[(lr)].append(acc)
    else:
        for lr in LEARNING_RATE:
            for margin in MARGIN:
                print('Learning Rate = ', lr, ', Margin = ',margin)
                Accuracy_Dict[(lr,margin)] = []
                for nFold in list(range(len(y_fold))):
                    print('Fold #',nFold+1)
                    excludeIndex = len(y_fold)-1-nFold
                    train_X, train_Y, test_X, test_Y = createFold(x_fold,y_fold,excludeIndex)
                    w, b, hist = train(train_X,train_Y,epochs=epochs,lr=lr,decay=decay,avg=avg,pocket=pocket,margin=margin)
                    acc = accuracy(test_X,test_Y,w,b)
                    Accuracy_Dict[(lr,margin)].append(acc)
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

def generate_data(num_samples):
    size = num_samples // 2
    x1 = np.random.multivariate_normal([0, 0], np.eye(2), size)
    y1 = -np.ones(size).astype(int)
    x2 = np.random.multivariate_normal([3, 3], np.eye(2), size)
    y2 = np.ones(size).astype(int)

    X = np.vstack((x1, x2))
    y = np.append(y1, y2)

    return X, y

def plot(x, y):
    fig = plt.figure(figsize = (7, 5), dpi = 100, facecolor = 'w')
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='black', cmap=cmap)
    plt.show()

def predict(X, w, b, value=0):
    if value == 0:
        y = np.sign(np.sum(w*X,-1+np.ndim(w*X))+b)
    else:
        y = np.sum(w*X,-1+np.ndim(w*X))+b
    return y

def shuffle_arrays(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train(X_train, y_train, epochs=10, lr=0.01, decay=None, avg=None, pocket=None, margin=None):
    if margin != None:
        return MarginPerceptron(X_train,y_train,epochs=epochs,lr=lr,margin=margin)
    elif pocket == 1:
        return PocketPerceptron(X_train,y_train,epochs=epochs,lr=lr)
    elif avg == 1:
        return AveragePerceptron(X_train,y_train,epochs=epochs,lr=lr)
    elif decay == 1:
        return DecayPerceptron(X_train,y_train,epochs=epochs,lr=lr)
    else:
        return SimplePerceptron(X_train,y_train,epochs=epochs,lr=lr)

def SimplePerceptron(X_train,y_train,epochs=10,lr=0.01):
    hist = History(epochs)
    w = np.random.uniform(0, 1, size=X_train.shape[1]) #initialize w
    b = 0 #initialize bias
    for e in list(range(epochs)):
        for i in list(range(len(y_train))):
            if predict(X_train[i],w,b) != y_train[i]:
                (w,b)=update(X_train[i],y_train[i],w,b,lr)
                acc = accuracy(X_train,y_train,w,b)
                hist.store(X_train[i],y_train[i],w,b,acc,e)
    return w, b, hist

def DecayPerceptron(X_train,y_train,epochs=10,lr=0.01):
    hist = History(epochs)
    w = np.random.uniform(0, 1, size=X_train.shape[1]) #initialize w
    b = 0 #initialize bias
    count = 0
    for e in list(range(epochs)):
        for i in list(range(len(y_train))):
            if predict(X_train[i],w,b) != y_train[i]:
                (w,b)=update(X_train[i],y_train[i],w,b,lr)
                acc = accuracy(X_train,y_train,w,b)
                hist.store(X_train[i],y_train[i],w,b,acc,e)
            count = count + .00000001
            lr = lr / (1 + count)
    return w, b, hist

def AveragePerceptron(X_train,y_train,epochs=10,lr=0.01):
    hist = History(epochs)
    w = np.random.uniform(0, 1, size=X_train.shape[1]) #initialize w
    b = 0 #initialize bias
    wa = w
    ba = b
    count = 0
    for e in list(range(epochs)):
        for i in list(range(len(y_train))):
            if predict(X_train[i],w,b) != y_train[i]:
                (w,b)=update(X_train[i],y_train[i],w,b,lr)
                acc = accuracy(X_train,y_train,wa,ba)
                hist.store(X_train[i],y_train[i],wa,ba,acc,e)
            count = count + 1
            wa = wa + w
            ba = ba + b
    wa = wa/count
    ba = ba/count
    return wa, ba, hist

def PocketPerceptron(X_train,y_train,epochs=10,lr=0.01):
    hist = History(epochs)
    w = np.random.uniform(0, 1, size=X_train.shape[1]) #initialize w
    b = 0 #initialize bias
    wp = w
    bp = b
    best_accuracy = accuracy(X_train,y_train,w,b)
    for e in list(range(epochs)):
        for i in list(range(len(y_train))):
            if predict(X_train[i],w,b) != y_train[i]:
                (w,b)=update(X_train[i],y_train[i],w,b,lr)
                acc = accuracy(X_train,y_train,w,b)
                if acc > best_accuracy:
                    best_accuracy = acc
                    wp = w
                    bp = b
                hist.store(X_train[i],y_train[i],wp,bp,best_accuracy,e)
    return wp, bp, hist

def MarginPerceptron(X_train,y_train,epochs=10,lr=0.01,margin=0.01):
    hist = History(epochs)
    w = np.random.uniform(0, 1, size=X_train.shape[1]) #initialize w
    b = 0 #initialize bias
    count = 0
    for e in list(range(epochs)):
        for i in list(range(len(y_train))):
            if predict(X_train[i],w,b) != y_train[i]:
                (w,b)=update(X_train[i],y_train[i],w,b,lr)
                acc = accuracy(X_train,y_train,w,b)
                hist.store(X_train[i],y_train[i],w,b,acc,e)
            elif (abs(predict(X_train[i],w,b,value=1)) < margin):
                (w,b)=update(X_train[i],y_train[i],w,b,lr,margin=1)
                acc = accuracy(X_train,y_train,w,b)
                hist.store(X_train[i],y_train[i],w,b,acc,e)
            count = count + .00000001
            lr = lr / (1 + count)
    return w, b, hist

def update(x, y, w, b, lr, margin=0):
    s = np.sign(y)
    w_new = w + s*lr*x
    b_new = round(b + s*lr,2)
    return (w_new,b_new)


def visualize(X, Y, epoch_hist):
    fig = plt.figure(figsize = (7, 5), dpi = 100, facecolor = 'w')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor='black', cmap=cmap)
    plots = []
    for e in epoch_hist:
        epoch_values = epoch_hist[e]
        w_hist = epoch_values['w_hist']
        b_hist = epoch_values['b_hist']
        acc_hist = epoch_values['acc_hist']
        point_hist = epoch_values['point_hist']
        for i in range(len(w_hist)):
            w, b = w_hist[i], b_hist[i]
            acc = acc_hist[i]
            if i+1 < len(point_hist):
                p_x, p_y = point_hist[i+1]
            else:
                p_x, p_y = point_hist[i]
            Z = predict(np.c_[xx.ravel(), yy.ravel()], w, b)
            Z = Z.reshape(xx.shape)
            plot =  plt.contourf(xx, yy, Z, cmap=cmap_region)
            text = f'Epoch: {e + 1} - Accuracy: {round(acc, 3)}'
            te = plt.text(90, 90, text)
            an = plt.annotate(text, xy=(0.3, 1.05), xycoords='axes fraction')
            points = plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor='black', cmap=cmap)
            c_idx = 1 if p_y == 1 else 0
            if i+1 < len(point_hist):
                p = plt.scatter(x=p_x[0], y=p_x[1], s=100, c=colors[c_idx], edgecolor='black')
                plots.append(plot.collections + [te, an, points, p])
            else:
                plots.append(plot.collections + [te, an, points])
    return animation.ArtistAnimation(fig, plots, repeat=False, blit=False, interval=500)

def getWordIndex(W):
    Most_Common = []
    Least_Common = []
    W=W.tolist()
    SortedList = copy.deepcopy(W)
    SortedList.sort()
    for word in SortedList[:10]:
        Least_Common.append(W.index(word))

    for word in SortedList[-10:]:
        Most_Common.append(W.index(word))
    return(Least_Common,Most_Common)

################################################################################
#
# def getParameterCombinations(HyperParameters):
#     parameterLabels = []
#     for parameter in HyperParameters.keys():
#          parameterLabels.append(parameter)
#     Combinations = []
#     for i in list(range(len(parameterLabels))):
#         if i == 0:
#             newCombinations = []
#             for value in HyperParameters[parameterLabels[i]]:
#                 newCombinations.append([value])
#         else:
#             newCombinations = []
#             for c in Combinations:
#                 for value in HyperParameters[parameterLabels[i]]:
#                     newCombinations.append(c+[value])
#         Combinations = copy.deepcopy(newCombinations)
#     return parameterLabels, Combinations
