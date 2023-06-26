from libsvm import *
import numpy as np
from ML_magic import *

###############################################################################
# IMPORTING DATA ##############################################################
###############################################################################
# Semeion Data ################################################################

print('Importing Semeion Data ...')
# TRAINING DATA
semeion_x_train, semeion_y_train, semeion_num_features_train = read_libsvm('data/data_semeion/hand_data_train')
semeion_x_train = addBias(semeion_x_train.toarray())

# TESTING DATA
semeion_x_test, semeion_y_test, semeion_num_features_test = read_libsvm('data/data_semeion/hand_data_test')
semeion_x_test = addBias(semeion_x_test.toarray())
# FOLD 1-5
semeion_x_fold1, semeion_y_fold1, semeion_num_features1 = read_libsvm('data/data_semeion/folds/fold1')
semeion_x_fold1 = addBias(semeion_x_fold1.toarray())

semeion_x_fold2, semeion_y_fold2, semeion_num_features2 = read_libsvm('data/data_semeion/folds/fold2')
semeion_x_fold2 = addBias(semeion_x_fold2.toarray())

semeion_x_fold3, semeion_y_fold3, semeion_num_features3 = read_libsvm('data/data_semeion/folds/fold3')
semeion_x_fold3 = addBias(semeion_x_fold3.toarray())

semeion_x_fold4, semeion_y_fold4, semeion_num_features4 = read_libsvm('data/data_semeion/folds/fold4')
semeion_x_fold4 = addBias(semeion_x_fold4.toarray())

semeion_x_fold5, semeion_y_fold5, semeion_num_features5 = read_libsvm('data/data_semeion/folds/fold5')
semeion_x_fold5 = addBias(semeion_x_fold5.toarray())

semeion_x_fold = [semeion_x_fold1, semeion_x_fold2, semeion_x_fold3, semeion_x_fold4, semeion_x_fold5]
semeion_y_fold = [semeion_y_fold1, semeion_y_fold2, semeion_y_fold3, semeion_y_fold4, semeion_y_fold5]

# Madelon Data ################################################################
print('Importing Madelon Data ...')
# TRAINING DATA
madelon_x_train, madelon_y_train, madelon_num_features_train = read_libsvm('data/data_madelon/madelon_data_train')
madelon_x_train = addBias(madelon_x_train.toarray())
madelon_x_train = binarize(madelon_x_train)
# TESTING DATA
madelon_x_test, madelon_y_test, madelon_num_features_test = read_libsvm('data/data_madelon/madelon_data_test')
madelon_x_test = addBias(madelon_x_test.toarray())
madelon_x_test = binarize(madelon_x_test)
# FOLD 1-5
madelon_x_fold1, madelon_y_fold1, madelon_num_features1 = read_libsvm('data/data_madelon/folds/fold1')
madelon_x_fold1 = addBias(madelon_x_fold1.toarray())
madelon_x_fold1 = binarize(madelon_x_fold1)

madelon_x_fold2, madelon_y_fold2, madelon_num_features2 = read_libsvm('data/data_madelon/folds/fold2')
madelon_x_fold2 = addBias(madelon_x_fold2.toarray())
madelon_x_fold2 = binarize(madelon_x_fold2)

madelon_x_fold3, madelon_y_fold3, madelon_num_features3 = read_libsvm('data/data_madelon/folds/fold3')
madelon_x_fold3 = addBias(madelon_x_fold3.toarray())
madelon_x_fold3 = binarize(madelon_x_fold3)

madelon_x_fold4, madelon_y_fold4, madelon_num_features4 = read_libsvm('data/data_madelon/folds/fold4')
madelon_x_fold4 = addBias(madelon_x_fold4.toarray())
madelon_x_fold4 = binarize(madelon_x_fold4)

madelon_x_fold5, madelon_y_fold5, madelon_num_features5 = read_libsvm('data/data_madelon/folds/fold5')
madelon_x_fold5 = addBias(madelon_x_fold5.toarray())
madelon_x_fold5 = binarize(madelon_x_fold5)

madelon_x_fold = [madelon_x_fold1, madelon_x_fold2, madelon_x_fold3, madelon_x_fold4, madelon_x_fold5]
madelon_y_fold = [madelon_y_fold1, madelon_y_fold2, madelon_y_fold3, madelon_y_fold4, madelon_y_fold5]

###############################################################################
# SUB-GRADIENT STOCHASTIC GRADIENT DESCENT ####################################
###############################################################################
# SEMEION #####################################################################
# CROSS VALIDATION ############################################################
# Hyper-Parameters
print(':::: SGD - Semeion ::::::::::::::::::::::::::::::::::::::::::::::::')
print('(Semeion) Cross-Validation with SGD')
LEARNING_RATE = [10, 1, .1, .01, .001, .0001]
LOSS_TRADEOFF = [10, 1, .1, .01, .001, .0001]
Accuracies = crossValidation("sgd",semeion_x_fold,semeion_y_fold,HP1=LEARNING_RATE,HP2=LOSS_TRADEOFF)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Semeion) Best Hyperparameters:',bestHP)
print('(Semeion) Average Cross-Validation Accuracy: ',bestACC)
# TRAINING ####################################################################
print('(Semeion) Training SGD...')
W = sgd(semeion_x_train,semeion_y_train,bestHP[0],bestHP[1])
# TESTING #####################################################################
print('(Semeion) Training Accuracy:',accuracy(semeion_x_train,semeion_y_train,W))
print('(Semeion) Testing Accuracy:',accuracy(semeion_x_test,semeion_y_test,W))
print('(Semeion) Finished with SGD')

# MADELON #####################################################################
# CROSS VALIDATION ############################################################
# Hyper-Parameters
print(':::: SGD - Madelon ::::::::::::::::::::::::::::::::::::::::::::::::')
print('(Madelon) Cross-Validation with SGD')
LEARNING_RATE = [10, 1, .1, .01, .001, .0001]
LOSS_TRADEOFF = [10, 1, .1, .01, .001, .0001]
Accuracies = crossValidation("sgd",madelon_x_fold,madelon_y_fold,HP1=LEARNING_RATE,HP2=LOSS_TRADEOFF)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Madelon) Best Hyperparameters:',bestHP)
print('(Madelon) Average Cross-Validation Accuracy: ',bestACC)
# TRAINING ####################################################################
print('(Madelon) Training SGD...')
W = sgd(madelon_x_train,madelon_y_train,bestHP[0],bestHP[1])
# TESTING #####################################################################
print('(Madelon) Training Accuracy:',accuracy(madelon_x_train,madelon_y_train,W))
print('(Madelon) Testing Accuracy:',accuracy(madelon_x_test,madelon_y_test,W))
print('(Madelon) Finished with SGD')

###############################################################################
# NAIVE BAYES #################################################################
###############################################################################
print(':::: Naive Bayes - Semeion ::::::::::::::::::::::::::::::::::::::::::::::::')
# CROSS VALIDATION ############################################################
# Hyper-Parameters
print('(Semeion) Cross-Validation with Naive Bayes')
SMOOTHING_TERM = [2.0, 1.5, 1.0, 0.5]
Accuracies = crossValidation("naive_bayes",semeion_x_fold,semeion_y_fold,HP1=SMOOTHING_TERM)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Semeion) Best Smoothing Term: ',bestHP)
print('(Semeion) Average Cross-Validation Accuracy: ',bestACC)
# TRAINING ####################################################################
print('(Semeion) Training NV...')
W = naive_bayes(semeion_x_train,semeion_y_train,bestHP)
# TESTING #####################################################################
print('(Semeion) Training Accuracy:',accuracy(semeion_x_train,semeion_y_train,W))
print('(Semeion) Testing Accuracy:',accuracy(semeion_x_test,semeion_y_test,W))
print('(Semeion) Finished with Naive Bayes')

print(':::: Naive Bayes - Madelon ::::::::::::::::::::::::::::::::::::::::::::::::')
# CROSS VALIDATION ############################################################
# Hyper-Parameters
print('(Madelon) Cross-Validation with Naive Bayes')
SMOOTHING_TERM = [2.0, 1.5, 1.0, 0.5]
Accuracies = crossValidation("naive_bayes",madelon_x_fold,madelon_y_fold,HP1=SMOOTHING_TERM)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Madelon) Best Smoothing Term: ',bestHP)
print('(Madelon) Average Cross-Validation Accuracy: ',bestACC)
# TRAINING ####################################################################
print('(Madelon) Training NV...')
W = naive_bayes(madelon_x_train,madelon_y_train,bestHP)
# TESTING #####################################################################
print('(Madelon) Training Accuracy:',accuracy(madelon_x_train,madelon_y_train,W))
print('(Madelon) Testing Accuracy:',accuracy(madelon_x_test,madelon_y_test,W))
print('(Madelon) Finished with Naive Bayes')

###############################################################################
# RANDOM FOREST ###############################################################
###############################################################################
# Hyper-Parameters
print(':::: Random Trees - Semeion ::::::::::::::::::::::::::::::::::::::::::::::::')
print('(Semeion) Cross Validation for Random Trees')
FOREST_SIZE = [10, 50, 100]
Accuracies = crossValidation("random_forest",semeion_x_fold,semeion_y_fold,HP1=FOREST_SIZE)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Semeion) Best Forest Size: ',bestHP)
print('(Semeion) Average Cross-Validation Accuracy',bestACC)
# TRAINING ####################################################################
print('(Semeion) Training Random Forest...')
forestS = random_forest(semeion_x_train,semeion_y_train,SampleSize=100,FeatureSize=50,k=bestHP)
# TESTING #####################################################################
print('(Semeion) Training Accuracy:',accuracy_forest(semeion_x_train,semeion_y_train,forestS))
print('(Semeion) Testing Accuracy:',accuracy_forest(semeion_x_test,semeion_y_test,forestS))
print('(Semeion) Finished with Random Trees')

# Hyper-Parameters
print(':::: Random Trees - Madelon ::::::::::::::::::::::::::::::::::::::::::::::::')
print('(Madelon) Cross Validation for Random Trees')
FOREST_SIZE = [10, 50, 100]
Accuracies = crossValidation("random_forest",madelon_x_fold,madelon_y_fold,HP1=FOREST_SIZE)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Madelon) Best Forest Size: ',bestHP)
print('(Madelon) Average Cross-Validation Accuracy',bestACC)
# TRAINING ####################################################################
print('(Madelon) Training Random Forest...')
forestM = random_forest(madelon_x_train,madelon_y_train,SampleSize=100,FeatureSize=50,k=bestHP)
# TESTING #####################################################################
print('(Madelon) Training Accuracy:',accuracy_forest(madelon_x_train,madelon_y_train,forestM))
print('(Madelon) Testing Accuracy:',accuracy_forest(madelon_x_test,madelon_y_test,forestM))
print('(Madelon) Finished with Random Trees')

###############################################################################
# SVM OVER TREES ##############################################################
###############################################################################

print(':::: SVM OVER TREES - SEMEION :::::::::::::::::::::::::::::::::::::::::')
# Hyper-Parameters
print('(Semeion) Cross Validation for SVM over Trees')
LEARNING_RATE = [1, .1, .01, .001, .0001, .00001]
LOSS_TRADEOFF = [10, 1, .1, .01, .001, .0001, .00001]
Accuracies = crossValidation("svm",semeion_x_fold,semeion_y_fold,HP1=LEARNING_RATE,HP2=LOSS_TRADEOFF,forest=forestS)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Semeion) Best Hyperparameters: ',bestHP)
print('(Semeion) Average Cross-Validation Accuracy',bestACC)

print('(Semeion) Training SVM over Trees ... ')
W = svm(semeion_x_train,semeion_y_train,forestS,bestHP[0],bestHP[1])
print('(Semeion) Finished Training...')

print('(Semeion) Training Accuracy:',accuracy_svm(semeion_x_train,semeion_y_train,W,forestS))
print('(Semeion) Testing Accuracy:',accuracy_svm(semeion_x_test,semeion_y_test,W,forestS))
print('(Semeion) Finished with SVM over Trees')

print(':::: SVM OVER TREES - MADELON :::::::::::::::::::::::::::::::::::::::::')
# Hyper-Parameters
print('(Madelon) Cross Validation for SVM over Trees')
LEARNING_RATE = [1, .1, .01, .001, .0001, .00001]
LOSS_TRADEOFF = [10, 1, .1, .01, .001, .0001, .00001]
Accuracies = crossValidation("svm",madelon_x_fold,madelon_y_fold,HP1=LEARNING_RATE,HP2=LOSS_TRADEOFF,forest=forestM)
bestHP, bestACC = findBestHyperParameters(Accuracies)
print('(Madelon) Best Hyperparameters: ',bestHP)
print('(Madelon) Average Cross-Validation Accuracy',bestACC)

print('(Madelon) Training SVM over Trees ... ')
W = svm(madelon_x_train,madelon_y_train,forestM,bestHP[0],bestHP[1])
print('(Madelon) Finished Training...')

print('(Madelon) Training Accuracy:',accuracy_svm(madelon_x_train,madelon_y_train,W,forestM))
print('(Madelon) Testing Accuracy:',accuracy_svm(madelon_x_test,madelon_y_test,W,forestM))
print('(Madelon) Finished with SVM over Trees')

##########################################################################
# Sandbox ################################################################
##########################################################################
