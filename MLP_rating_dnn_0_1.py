################################################################################
#
# M A I N
#
# ################################################################################

#cd  /media/user/_home1/apps/python/DL/Agri
#cd  D:/Apps/Python/DL/Agri
import math
import numpy as np
import matplotlib.pyplot as plt
import csv 
import pandas as pd  
from sklearn.neural_network import MLPClassifier
%matplotlib inline


  def load_dataset():
  #
  # load, split & shuffle 40.000 tuples dataset
  #
    Dataset = []
#    with open('C:\\Apps\\python\\DL\\AuditLab\\data\\DatasetNuovoConAAeB.csv','rt') as csvfile: 
    with open('data/DatasetNuovoConAAeBFiltrato.csv','rt') as csvfile: 
        reader = csv.reader(csvfile, delimiter=';') 
        for row in reader:
            Dataset.append(row)
    X_Dataset_orig = np.array(np.array(Dataset[1:])[:,3:], float)
    Y_Dataset_orig = np.where( np.array(np.array(Dataset[1:])[:,2]) == 'AA', 1, -1)
    #Shuffle (X, Y)
    seed=0
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X_Dataset_orig.shape[0]                  # number of training examples
    permutation = list(np.random.permutation(m))
    X_Dataset_shuffled = X_Dataset_orig[permutation, :]
    Y_Dataset_shuffled = Y_Dataset_orig[permutation]
  
    N = Y_Dataset_orig.shape[0]
    N1 = N - math.floor(N/3)
    X_Dataset_train = X_Dataset_shuffled[:N1,:]
    Y_Dataset_train = Y_Dataset_shuffled[:N1]
    X_Dataset_test = X_Dataset_shuffled[N1:,:]
    Y_Dataset_test = Y_Dataset_shuffled[N1:]

    #Normalize Data
    epsilon = 1e-8
    for i in range(N1):
      X_Dataset_train[i,:] = (X_Dataset_train[i,:] - np.mean(X_Dataset_train, axis = 0))/(np.std(X_Dataset_train, axis = 0) + epsilon)  
    for i in range(N-N1):
      X_Dataset_test[i,:] = (X_Dataset_test[i,:] - np.mean(X_Dataset_test, axis = 0))/(np.std(X_Dataset_test, axis = 0) + epsilon)  
    
    classes = np.array(range(2), int)
    return X_Dataset_train, Y_Dataset_train, X_Dataset_test, Y_Dataset_test, classes


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train = X_train_orig
X_test = X_test_orig
Y_train = Y_train_orig
Y_test = Y_test_orig

# FIT Model
clf = MLPClassifier(solver='adam', alpha=1e-5, learning_rate_init=0.01, hidden_layer_sizes=(20, 2), random_state=1)
clf.fit(X_train, Y_train)

#PREDICT test set
test1_label=clf.predict(X_test)


j = 0 
for i in range(Y_test.size):
#  print('predict:', test1_label[i], 'test:', Y_test[i])
  if test1_label[i] == Y_test[i]: 
    j += 1

acc = j / i
print("Accurance: ", acc)