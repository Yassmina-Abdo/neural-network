import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------
def sigmoid(s):
    s = 1/(1+np.exp(-s))
    return s
#----------------------------------------------------------------
def Hyperbolic(s):
    s = np.tanh(s)
    return s
#------------------------------------------------------------------
def Preprocessing():
    data = pd.read_csv('IrisData.txt')
    X = data.loc[:, ['X1','X2','X3','X4']]
    y = data.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    y1 = y[:50]
    y2 = y[50:100]
    y3 = y[100:]



    #--------- Feature 1 --------------------
    X1=np.array(X.iloc[:50, 0]) # 1
    np.random.shuffle(X1)
    X1=np.reshape(X1,(X1.shape[0],1))
    X_train1_1=X1[:30,0]
    X_test1_1=X1[30:50,0]
    X1 = np.array(X.iloc[50:100, 0]) # 2
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train1_2 = X1[:30, 0]
    X_test1_2 = X1[30:50, 0]
    X1 = np.array(X.iloc[100:, 0]) # 3
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train1_3 = X1[:30, 0]
    X_test1_3 = X1[30:50, 0]
    # --------- Feature 2 --------------------
    X1 = np.array(X.iloc[:50, 1]) # 1
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train2_1 = X1[:30, 0]
    X_test2_1 = X1[30:50, 0]
    X1 = np.array(X.iloc[50:100, 1]) # 2
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train2_2 = X1[:30, 0]
    X_test2_2 = X1[30:50, 0]
    X1 = np.array(X.iloc[100:, 1]) # 3
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train2_3 = X1[:30, 0]
    X_test2_3 = X1[30:50, 0]
    # --------- Feature 3 --------------------
    X1 = np.array(X.iloc[:50, 2]) # 1
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train3_1 = X1[:30, 0]
    X_test3_1 = X1[30:50, 0]
    X1 = np.array(X.iloc[50:100, 2]) # 2
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train3_2 = X1[:30, 0]
    X_test3_2 = X1[30:50, 0]
    X1 = np.array(X.iloc[100:, 2]) # 3
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train3_3 = X1[:30, 0]
    X_test3_3 = X1[30:50, 0]
    # --------- Feature 4 --------------------
    X1 = np.array(X.iloc[:50, 3]) # 1
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train4_1 = X1[:30, 0]
    X_test4_1 = X1[30:50, 0]
    X1 = np.array(X.iloc[50:100, 3]) # 2
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train4_2 = X1[:30, 0]
    X_test4_2 = X1[30:50, 0]
    X1 = np.array(X.iloc[100:, 3]) # 3
    X1=np.reshape(X1,(X1.shape[0],1))
    np.random.shuffle(X1)
    X_train4_3 = X1[:30, 0]
    X_test4_3 = X1[30:50, 0]
    #-----------------------------------------------------------
    X1_train = np.concatenate((X_train1_1,X_train1_2,X_train1_3))
    X2_train = np.concatenate((X_train2_1,X_train2_2,X_train2_3))
    X3_train = np.concatenate((X_train3_1, X_train3_2, X_train3_3))
    X4_train = np.concatenate((X_train4_1, X_train4_2, X_train4_3))
    X1_train=np.reshape(X1_train,(X1_train.shape[0],1))
    X2_train=np.reshape(X2_train,(X2_train.shape[0],1))
    X3_train=np.reshape(X3_train,(X3_train.shape[0],1))
    X4_train=np.reshape(X4_train,(X4_train.shape[0],1))
    X_train = np.concatenate((X1_train, X2_train, X3_train, X4_train),axis=1)
    #--------------------------------------------------------------
    X1_test  = np.concatenate((X_test1_1,X_test1_2,X_test1_3))
    X2_test  = np.concatenate((X_test2_1, X_test2_2, X_test2_3))
    X3_test  = np.concatenate((X_test3_1, X_test3_2, X_test3_3))
    X4_test  = np.concatenate((X_test4_1, X_test4_2, X_test4_3))
    X1_test=np.reshape(X1_test,(X1_test.shape[0],1))
    X2_test=np.reshape(X2_test,(X2_test.shape[0],1))
    X3_test=np.reshape(X3_test,(X3_test.shape[0],1))
    X4_test=np.reshape(X4_test,(X4_test.shape[0],1))
    X_test = np.concatenate((X1_test, X2_test, X3_test, X4_test),axis=1)
    #--------------------------------------------------------------
    y1_tarin = y1[:30]
    y2_train = y2[:30]
    y3_train = y3[:30]
    y_train = np.concatenate((y1_tarin, y2_train, y3_train))
    y_train=np.reshape(y_train,(y_train.shape[0],1))
    #---------------------------------------------------------------
    y1_test = y1[30:50]
    y2_test = y2[30:50]
    y3_test = y3[30:50]
    y_test=np.concatenate((y1_test,y2_test,y3_test))
    y_test=np.reshape(y_test,(y_test.shape[0],1))
    
    min_max_sc = MinMaxScaler()
    X_train = min_max_sc.fit_transform(X_train)
    X_test = min_max_sc.transform(X_test)

    return X_train, X_test, y_train, y_test
#-------------------------------------------------------------------

def initialize_parameters(Neurons_List):
    parameters = {}

    for i in range(1, len(Neurons_List)):
        parameters["W" + str(i)] = np.random.randn(Neurons_List[i - 1], Neurons_List[i])*0.01
        parameters["b" + str(i)] = np.random.randn(Neurons_List[i], 1)*0.01
    parameters["W" + str(len(Neurons_List))] = np.random.randn(Neurons_List[len(Neurons_List)-1], 3)*0.01
    parameters["b" + str(len(Neurons_List))] = np.random.randn(3, 1)*0.01
    return parameters





def evaluate_NN(Epoch,Rate,Hidden_layer,Neurons,Activation_Function,bais_checked):

    results = {}
    
    X_train, X_test, y_train, y_test = Preprocessing()
    Neurons_List = list(map(int ,Neurons.split(',')))
    Neurons_List.insert(0, 4)  # Append Input Neurons With Hidden N
    
    myparameters=initialize_parameters(Neurons_List)
    




    #-------------------------------------------------------------------------------------------------
    for i in range(Epoch):
        results["A0"] = X_train.copy()
        for h in range(1,Hidden_layer+2):
    
            if(bais_checked==True):
                results["A"+str(h)] = FeedFarward(results["A"+str(h-1)], myparameters["W"+str(h)],Activation_Function,myparameters["b"+str(h)],bais_checked)
            else:
                results["A"+str(h)] = FeedFarward(results["A"+str(h-1)], myparameters["W" + str(h)],Activation_Function,bais_checked)
    
    
    
        myparameters = BackFarward(X_train, y_train,Rate,Activation_Function,Hidden_layer,bais_checked,myparameters, results)
        



    #-----------------------------------------------------------------------------------------------------------
    results["A0"] = X_test.copy()
    for h in range(1,Hidden_layer+2):

           if(bais_checked==True):
               results["A"+str(h)] = FeedFarward(results["A"+str(h-1)], myparameters["W"+str(h)],Activation_Function,myparameters["b"+str(h)],bais_checked)
               
           else:
               results["A"+str(h)] = FeedFarward(results["A"+str(h-1)], myparameters["W" + str(h)],Activation_Function,bais_checked)

   
    
    y_pred=np.zeros((60,1))
    for i in range(len(results["A"+str(Hidden_layer+1)])):
        if(results["A"+str(Hidden_layer+1)][i,0]>= results["A"+str(Hidden_layer+1)][i,1] and 
            results["A"+str(Hidden_layer+1)][i,0]>= results["A"+str(Hidden_layer+1)][i,2]):
            y_pred[i,0] = 0
        elif(results["A"+str(Hidden_layer+1)][i,1]>= results["A"+str(Hidden_layer+1)][i,0] and
           results["A"+str(Hidden_layer+1)][i,1]>= results["A"+str(Hidden_layer+1)][i,2]):
            y_pred[i,0] = 1
        elif(results["A"+str(Hidden_layer+1)][i,2]>= results["A"+str(Hidden_layer+1)][i,0] and
           results["A"+str(Hidden_layer+1)][i,2]>= results["A"+str(Hidden_layer+1)][i,1]):
            y_pred[i,0] = 2

    y_test = np.reshape(y_test, (60, 1))
    CM = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return CM,acc
#-----------------------------------------------------------------------------


def BackFarward(X_train,y_train,Rate,Activation,Hidden,bais,parameters, results):
    errors = {}
    output_err = (y_train - results["A"+str(Hidden+1)])

    if (Activation == "Sigmoid"):
        errors["dz" + str(Hidden+1)] = output_err * (results["A"+str(Hidden+1)] * (1 - results["A"+str(Hidden+1)]))
        if(bais==True):
            errors["dw"+str(Hidden+1)] = np.dot(results["A"+str(Hidden)].T, errors["dz" + str(Hidden+1)])
            errors["db" + str(Hidden + 1)] =  np.sum(errors["dz" + str(Hidden+1)],axis=0,keepdims=True)
        else:
            errors["dw" + str(Hidden + 1)] = np.dot(results["A"+str(Hidden)].T, errors["dz" + str(Hidden+1)])
    else:
        errors["dz" + str(Hidden+1)] = output_err * (1-results["A"+str(Hidden+1)]**2)
        if (bais == True):
            errors["dw"+str(Hidden+1)] = np.dot(results["A"+str(Hidden)].T, errors["dz" + str(Hidden+1)])
            errors["db" + str(Hidden + 1)] = np.sum(errors["dz" + str(Hidden+1)],axis=0,keepdims=True)
        else:
            errors["dw" + str(Hidden + 1)] = np.dot(results["A"+str(Hidden)].T, errors["dz" + str(Hidden+1)])


    for i in range(Hidden,0,-1):
        if (Activation == "Sigmoid"):
            dz = np.dot(errors["dz"+str(i+1)], parameters["W"+str(i+1)].T) * (results["A"+str(i)] * (1 - results["A" + str(i)]))
            errors["dz"+str(i)] = dz
            if(bais==True):
                if(i-1==0):
                    errors["dw" + str(i)] = np.dot(X_train.T, dz)
                else:
                    errors["dw" + str(i)] = np.dot(results["A"+str(i-1)].T, dz)
                errors["db" + str(i)] =  np.sum(dz,axis=0,keepdims=True)
            else:
                if(i-1==0):
                    errors["dw" + str(i)] = np.dot(X_train.T, dz)
                else:
                    errors["dw" + str(i)] = np.dot(results["A"+str(i-1)].T, dz)
                
        else:
            dz = np.dot(errors["dz"+str(i+1)], parameters["W"+str(i+1)].T) *(1 - results["A" + str(i)]**2)
            errors["dz"+str(i)] = dz
            if(bais==True):
                if(i-1==0):
                    errors["dw" + str(i)] = np.dot(X_train.T, dz)
                else:
                    errors["dw" + str(i)] = np.dot(results["A"+str(i-1)].T, dz)
                errors["db" + str(i)] =  np.sum(dz,axis=0,keepdims=True)
            else:
                if(i-1==0):
                    errors["dw" + str(i)] = np.dot(X_train.T, dz)
                else:
                    errors["dw" + str(i)] = np.dot(results["A"+str(i-1)].T, dz)

    # Update
    for i in range(1,Hidden+2):
        parameters["W"+str(i)] += (Rate*errors["dw" + str(i)])
        if(bais==True):
            parameters["b"+str(i)] += errors["db" + str(i)].T*Rate

    return  parameters





def FeedFarward(Input, W_Layer,Activation,Bais,Baischeck=False):
    if(Baischeck!=False):
        if(Bais.shape[0]!=1):
            s= Bais.T + np.dot(Input, W_Layer)
        else:
            s= Bais + np.dot(Input, W_Layer)
    else:
        s= np.dot(Input, W_Layer)

    if(Activation == "Sigmoid"):
        ActResult = sigmoid(s)
    else:
        ActResult = Hyperbolic(s)

    return  ActResult


