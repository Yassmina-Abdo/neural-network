import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#-----------------------------------------------------------
def signum(array):
    indx = 0
    for v in array:
        if v > 0:
            array[indx] = 1
        elif v == 0:
            array[indx] = 0
        else:
            array[indx] = -1
        indx += 1

    return array
#------------------------------------------------------------------
def Preprocessing(input1, input2, Class1, Class2):

    data = pd.read_csv('IrisData.txt')
    X = data.loc[:, [input1, input2]]
    y = data.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    #----------------------------------------------------------
    if (Class1 == 'Iris-setosa'and Class2=='Iris-versicolor'):
        for i in range(y.shape[0]):
            if y[i] == 0:
                y[i] = -1
    elif(Class1 == 'Iris-setosa'and Class2=='Iris-virginica'):
        for i in range(y.shape[0]):
            if y[i] == 0:
                y[i] = -1
            if y[i] == 2:
                y[i] = 1
    else:
        for i in range(y.shape[0]):
            if y[i] == 2:
                y[i] = -1
    #--------------------------------------------------------
    if Class1 == 'Iris-setosa':
        X1 =preprocessing.normalize( X.iloc[:50, :])
        X1=np.array(X1)
        np.random.shuffle(X1)
        X_train1=X1[:30,:]
        X_test1=X1[30:50,:]
        y1 = y[:50]
    elif Class1 == 'Iris-versicolor':
        X1 = preprocessing.normalize(X.iloc[50:100, :])
        X1 = np.array(X1)
        np.random.shuffle(X1)
        X_train1 = X1[:30, :]
        X_test1 = X1[30:50, :]
        y1 = y[50:100]
    else:
        X1 = preprocessing.normalize(X.iloc[100:, :])
        X1 = np.array(X1)
        np.random.shuffle(X1)
        X_train1 = X1[:30, :]
        X_test1 = X1[30:50, :]
        y1 = y[100:]
    #---------------------------------------------
    if Class2 == 'Iris-setosa':
        X2 = preprocessing.normalize(X.iloc[:50, :])
        X2 = np.array(X2)
        np.random.shuffle(X2)
        X_train2 = X2[:30, :]
        X_test2 = X2[30:50, :]
        y2 = y[:50]
    elif Class2 == 'Iris-versicolor':
        X2 = preprocessing.normalize(X.iloc[50:100, :])
        X2 = np.array(X2)
        np.random.shuffle(X2)
        X_train2 = X2[:30, :]
        X_test2 = X2[30:50, :]
        y2 = y[50:100]
    else:
        X2 = preprocessing.normalize(X.iloc[100:, :])
        X2 = np.array(X2)
        np.random.shuffle(X2)
        X_train2 = X2[:30, :]
        X_test2 = X2[30:50, :]
        y2 = y[100:]
    #----------------------------------------------------
    X_train=np.concatenate((X_train1,X_train2))
    X_test=np.concatenate((X_test1,X_test2))
    y_tarin1=y1[:30]
    y_train2=y2[:30]
    y_train=np.concatenate((y_tarin1,y_train2))
    y_test1 = y1[30:50]
    y_test2 = y2[30:50]
    y_test=np.concatenate((y_test1,y_test2))

    return X_train, X_test, y_train, y_test
#-------------------------------------------------------------------
def Perceptron(X, y, epochs=10, Rate=0.01, bais=True):

    W = np.random.uniform(low=-0.1, high=0.1, size=((X.shape[1], 1)))
    b = np.random.uniform(low=-0.1, high=0.1, size=((1, 1)))
    dw = np.zeros((X.shape[1], 1))
    db = 0
    y = np.reshape(y, (60, 1))
    for _ in range(epochs):
        if bais == True:
            y_pred = signum(np.dot(X, W) + b)
            loss = y-y_pred
            dw = np.dot(X.transpose(), loss)
            db = np.sum(loss)
            W = W + Rate * dw
            b = b + Rate * db
        else:
            y_pred = signum(np.dot(X, W))
            loss = y-y_pred
            dw += np.dot(X.transpose(), loss)
            W = W + Rate * dw

    return W, b, loss
#-------------------------------------------------------------------
def Adaline(X, y, epochs=10, Rate=0.01,mse=0.01, bais=True):

    W = np.random.uniform(low=-0.1, high=0.1, size=((X.shape[1], 1)))
    b = np.random.uniform(low=-0.1, high=0.1, size=((1, 1)))
    dw = np.zeros((X.shape[1], 1))
    db = 0
    y = np.reshape(y, (60, 1))
    while (100000):
       for _ in range(epochs):
          if bais == True:
             y_pred = np.dot(X, W)+b
             loss = y - y_pred
             dw = np.dot(X.transpose(), loss)
             db = np.sum(loss)
             W = W + Rate * dw
             b = b + Rate * db
          else:
             y_pred = np.dot(X, W)
             loss = y - y_pred
             dw += np.dot(X.transpose(), loss)
             W = W + Rate * dw
       for i in range(epochs):
           loss =y-y_pred

       theMse =  np.square(loss).mean()
       if (theMse < mse):
           break
       else:
           continue

    return W, b, loss
#--------------------------------------------------------------------
def train_Test_NN(features, classes, Rate, epochs,mse, bais):
    if (features == "X1 and X2"):
        X1 = 'X1'
        X2 = 'X2'
    elif (features == "X1 and X3"):
        X1 = 'X1'
        X2 = 'X3'
    elif (features == "X1 and X4"):
        X1 = 'X1'
        X2 = 'X4'
    elif (features == "X2 and X3"):
        X1 = 'X2'
        X2 = 'X3'
    elif (features == "X2 and X4"):
        X1 = 'X2'
        X2 = 'X4'
    else:
        X1 = 'X3'
        X2 = 'X4'
    #------------------------------------------------
    if (classes == "Iris-setosa and Iris-versicolor"):
        Class1 = 'Iris-setosa'
        Class2 = 'Iris-versicolor'
    elif (classes == "Iris-setosa and Iris-virginica"):
        Class1 = 'Iris-setosa'
        Class2 = 'Iris-virginica'
    else:
        Class1 = 'Iris-versicolor'
        Class2 = 'Iris-virginica'
    #------------------------------------------------
    if (bais == 1):
        newbais = True
    else:
        newbais = False
    # ------------------------------------------------
    X_train, X_test, y_train, y_test = Preprocessing(X1, X2, Class1, Class2)
    W, b, loss = Adaline(X_train, y_train, int(epochs), float(Rate),float(mse), newbais)
    y2_pred = signum(np.dot(X_test, W) + b)
    return W, b, loss, X_train, X_test, y_train, y_test, y2_pred
#--------------------------------------------------------------------
def draw_line(W,b,X):
    X_i1=min(X[:, 0])
    X_j1 = (-(W[0] * X_i1) - b) / W[1]
    X_j2=min(X[:, 1])
    X_i2 = (-(W[1] * X_j2) - b) / W[0]
    X_j3 = max(X[:, 1])
    X_i3 = (-(W[1] * X_j3) - b) / W[0]
    x_values = [X_i1,X_i2[0][0],X_i3[0][0]]
    y_values = [X_j1[0][0],X_j2,X_j3]
    X11 = X[:20, 0]
    X12 = X[:20,1]
    X21 = X[20:40, 0]
    X22 = X[20:40, 1]
    plt.figure('F')
    plt.scatter(X11, X12)
    plt.scatter(X21, X22)
    plt.plot(x_values, y_values)
    plt.show()
# --------------------------------------------------------------------
def evaluate_NN(y_pred, y_test):

    y_test = np.reshape(y_test, (40, 1))
    CM = confusion_matrix(y_true=y_test, y_pred=y_pred)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    return CM,acc

