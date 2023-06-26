from perceptron import *
from libsvm import *
import datetime

print('Start Time: ',datetime.datetime.now())

################################################################################
# IMPORTING DATA
################################################################################
# TRAINING DATA
x_train, y_train, num_features_train = read_libsvm('data_train')
x_train = x_train.toarray()

# TESTING DATA
x_test, y_test, num_features_test = read_libsvm('data_test')
x_test = x_test.toarray()

# FOLD 1-5
x_fold1, y_fold1, num_features1 = read_libsvm('fold1')
x_fold1 = x_fold1.toarray()

x_fold2, y_fold2, num_features2 = read_libsvm('fold2')
x_fold2 = x_fold2.toarray()

x_fold3, y_fold3, num_features3 = read_libsvm('fold3')
x_fold3 = x_fold3.toarray()

x_fold4, y_fold4, num_features4 = read_libsvm('fold4')
x_fold4 = x_fold4.toarray()

x_fold5, y_fold5, num_features5 = read_libsvm('fold5')
x_fold5 = x_fold5.toarray()

x_fold = [x_fold1, x_fold2, x_fold3, x_fold4, x_fold5]
y_fold = [y_fold1, y_fold2, y_fold3, y_fold4, y_fold5]

################################################################################
# THE PERCEPTIVE PERCEPTRON AND ITS VARIOUS VARIANTS!!! ########################
################################################################################

# SIMPLE PERCEPTRON ############################################################
print(':::::Simple Perceptron:::::::::::::::::::::::::::::::::::::::::::::::::')
print('Performing Cross-Validation...')
LEARNING_RATE = [1,0.1,0.01]
print('Hyper-Parameters: Learning Rate = ', LEARNING_RATE)
Accuracies = crossValidation(x_fold,y_fold,epochs=10,LEARNING_RATE=LEARNING_RATE)
bestHyperParameters, bestAccuracy = findBestHyperParameters(Accuracies)
print('Cross-Validation Completed')
print('\n')
print('Best Hyper-Parameters: Learning Rate = ',bestHyperParameters)
print('Cross-Validation Accuracy: ', round(bestAccuracy,3))
print('\n')
print('Training Perceptron...')
w, b, hist = train(x_train, y_train, epochs=20, lr=bestHyperParameters)
print('Training Completed')
bestClassifier = hist.getBestClassifier()
e=bestClassifier[0]
a=bestClassifier[1]
w=bestClassifier[2]
b=bestClassifier[3]
print('Best Classifier at epoch #',e,' with ',round(a,3),' accuracy')
word_list = getWordIndex(w)
print('Most Common Words: ', word_list[1])
print('Least Common Words: ', word_list[0])
print('Number of Updates: ', hist.updates)
train_acc = accuracy(x_train,y_train,w,b)
print('Training Set Accuracy: ', round(train_acc,3))
test_acc = accuracy(x_test,y_test,w,b)
print('Test Set Accuracy: ',round(test_acc,3))
print('Plotting Learning Curve ...')
hist.plotLearningCurve(save='simple_perceptron.png')

# DECAYING LEARNING RATE #######################################################
print(':::::Decay Perceptron:::::::::::::::::::::::::::::::::::::::::::::::::')
print('Performing Cross-Validation...')
LEARNING_RATE = [1,0.1,0.01]
print('Hyper-Parameters: Initial Learning Rate = ', LEARNING_RATE)
Accuracies = crossValidation(x_fold,y_fold,epochs=10,LEARNING_RATE=LEARNING_RATE,decay=1)
bestHyperParameters, bestAccuracy = findBestHyperParameters(Accuracies)
print('Cross-Validation Completed')
print('\n')
print('Best Hyper-Parameters: Inital Learning Rate = ',bestHyperParameters)
print('Cross-Validation Accuracy: ', round(bestAccuracy,3))
print('\n')
print('Training Perceptron...')
w, b, hist = train(x_train, y_train, epochs=20, lr=bestHyperParameters)
print('Training Completed')
bestClassifier = hist.getBestClassifier()
e=bestClassifier[0]
a=bestClassifier[1]
w=bestClassifier[2]
b=bestClassifier[3]
print('Best Classifier at epoch #',e,' with ',round(a,3),' accuracy')
word_list = getWordIndex(w)
print('Most Common Words: ', word_list[1])
print('Least Common Words: ', word_list[0])
print('Number of Updates: ', hist.updates)
train_acc = accuracy(x_train,y_train,w,b)
print('Training Set Accuracy: ', round(train_acc,3))
test_acc = accuracy(x_test,y_test,w,b)
print('Test Set Accuracy: ',round(test_acc,3))
print('Plotting Learning Curve ...')
hist.plotLearningCurve(save='decay_perceptron.png')

# # AVERAGED PERCEPTRON ##########################################################
print(':::::Averaged Perceptron:::::::::::::::::::::::::::::::::::::::::::::::::')
print('Performing Cross-Validation...')
LEARNING_RATE = [1,0.1,0.01]
print('Hyper-Parameters: Learning Rate = ', LEARNING_RATE)
Accuracies = crossValidation(x_fold,y_fold,epochs=10,LEARNING_RATE=LEARNING_RATE,avg=1)
bestHyperParameters, bestAccuracy = findBestHyperParameters(Accuracies)
print('Cross-Validation Completed')
print('\n')
print('Best Hyper-Parameters: Learning Rate = ',bestHyperParameters)
print('Cross-Validation Accuracy: ', round(bestAccuracy,3))
print('\n')
print('Training Perceptron...')
wa, ba, hist = train(x_train, y_train, epochs=20, lr=bestHyperParameters,avg=1)
print('Training Completed')
bestClassifier = hist.getBestClassifier()
e=bestClassifier[0]
a=bestClassifier[1]
wa=bestClassifier[2]
ba=bestClassifier[3]
print('Best Classifier at epoch #',e,' with ',round(a,3),' accuracy')
word_list = getWordIndex(w)
print('Most Common Words: ', word_list[1])
print('Least Common Words: ', word_list[0])
print('Number of Updates: ', hist.updates)
train_acc = accuracy(x_train,y_train,wa,ba)
print('Training Set Accuracy: ', round(train_acc,3))
test_acc = accuracy(x_test,y_test,wa,ba)
print('Test Set Accuracy: ',round(test_acc,3))
print('Plotting Learning Curve ...')
hist.plotLearningCurve(save='averaged_perceptron.png')

# POCKET PERCEPTRON ############################################################
print(':::::Pocket Perceptron:::::::::::::::::::::::::::::::::::::::::::::::::')
print('Performing Cross-Validation...')
LEARNING_RATE = [1,0.1,0.01]
print('Hyper-Parameters: Learning Rate = ', LEARNING_RATE)
Accuracies = crossValidation(x_fold,y_fold,epochs=10,LEARNING_RATE=LEARNING_RATE,pocket=1)
bestHyperParameters, bestAccuracy = findBestHyperParameters(Accuracies)
print('Cross-Validation Completed')
print('\n')
print('Best Hyper-Parameters: Learning Rate = ',bestHyperParameters)
print('Cross-Validation Accuracy: ', round(bestAccuracy,3))
print('\n')
print('Training Perceptron...')
wp, bp, hist = train(x_train, y_train, epochs=20, lr=bestHyperParameters,pocket=1)
print('Training Completed')
bestClassifier = hist.getBestClassifier()
e=bestClassifier[0]
a=bestClassifier[1]
wp=bestClassifier[2]
bp=bestClassifier[3]
print('Best Classifier at epoch #',e,' with ',round(a,3),' accuracy')
word_list = getWordIndex(w)
print('Most Common Words: ', word_list[1])
print('Least Common Words: ', word_list[0])
print('Number of Updates: ', hist.updates)
train_acc = accuracy(x_train,y_train,wp,bp)
print('Training Set Accuracy: ', round(train_acc,3))
test_acc = accuracy(x_test,y_test,wp,bp)
print('Test Set Accuracy: ',round(test_acc,3))
print('Plotting Learning Curve ...')
hist.plotLearningCurve(save='pocket_perceptron.png')

# MARGIN PERCEPTRON ############################################################
print(':::::Margin Perceptron:::::::::::::::::::::::::::::::::::::::::::::::::')
print('Performing Cross-Validation...')
LEARNING_RATE = [1,0.1,0.01]
MARGIN = [1,0.1,0.01]
print('Hyper-Parameters: Learning Rate = ', LEARNING_RATE,", Margin = ",MARGIN)
Accuracies = crossValidation(x_fold,y_fold,epochs=10,LEARNING_RATE=LEARNING_RATE,MARGIN=MARGIN)
bestHyperParameters, bestAccuracy = findBestHyperParameters(Accuracies)
print('Cross-Validation Completed')
print('\n')
print('Best Hyper-Parameters: Learning Rate = ',bestHyperParameters[0],', Margin = ',bestHyperParameters[1])
print('Cross-Validation Accuracy: ', round(bestAccuracy,3))
print('\n')
print('Training Perceptron...')
w, b, hist = train(x_train, y_train, epochs=20, lr=bestHyperParameters[0],margin=bestHyperParameters[1])
print('Training Completed')
bestClassifier = hist.getBestClassifier()
e=bestClassifier[0]
a=bestClassifier[1]
w=bestClassifier[2]
b=bestClassifier[3]
print('Best Classifier at epoch #',e,' with ',round(a,3),' accuracy')
word_list = getWordIndex(w)
print('Most Common Words: ', word_list[1])
print('Least Common Words: ', word_list[0])
print('Number of Updates: ', hist.updates)
train_acc = accuracy(x_train,y_train,w,b)
print('Training Set Accuracy: ', round(train_acc,3))
test_acc = accuracy(x_test,y_test,w,b)
print('Test Set Accuracy: ',round(test_acc,3))
print('Plotting Learning Curve ...')
hist.plotLearningCurve(save='margin_perceptron.png')


print('End Time: ',datetime.datetime.now())
