import copy
import time
import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
warnings.filterwarnings('ignore')

def reformat_date(date):
    return (datetime.datetime.strptime(date, '%d/%m/%Y'))

# X_train.shape = (1096, 65)
dim = 65

def NN1():
    # create model
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
def NN1_dropout():
    # create model
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def NN2():
    # create model
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def NN3():
    # create model
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def NN4():
    # create model
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def NN5():
    # create model
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Model_return(Return_data, y_pred):
    returns = copy.deepcopy(Return_data)
    current_value = 1
    for i in range(len(Return_data)):
        if y_pred[i] == 1:
            current_value = current_value*(1+Return_data[i])
            returns[i] = current_value
        else:
            returns[i] = current_value
    return returns

def Performance(Y_train, y_pred):
    results = []
    performance_funcs = [precision_score, recall_score, f1_score, roc_auc_score, accuracy_score]
    for func in performance_funcs:
        results.append(func(Y_train, y_pred))
    return results

def Undersampling(X_train, Y_train):
    nm = NearMiss()
    return nm.fit_resample(X_train, Y_train)

def Oversampling(X_train, Y_train):
    smote = SMOTE()
    return smote.fit_resample(X_train, Y_train)

def Plot_return(returns):
    df = returns.T
    df.columns = ['180_moving_window', '365_moving_window', '730_moving_window']
    return df.plot.line()

def Window_LogReg(X_train, Y_train, Return_train, window, moving, over, penalty='l2'):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        log = LogisticRegression(penalty, solver='lbfgs', max_iter=4000)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        log.fit(X, Y)
        y_pred[j:j+1] = log.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_pca(X_train, Y_train, Return_train, window, moving, over, n_components=2, penalty='l2'):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        pca = Pipeline([('pca', PCA(n_components)),('clf', LogisticRegression(penalty, solver='lbfgs', max_iter=4000))])
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        pca.fit(X, Y)
        y_pred[j:j+1] = pca.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_rf(X_train, Y_train, Return_train, window, moving, over, n=20, d=10):
    n_estimators = n
    max_depth = d
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=123,n_jobs=-1)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        rf.fit(X, Y)
        y_pred[j:j+1] = rf.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_KNN(X_train, Y_train, Return_train, window, moving, over, n_neighbors=25):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        KNN = KNeighborsClassifier(n_neighbors,n_jobs=-1)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        KNN.fit(X, Y)
        y_pred[j:j+1] = KNN.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_NN(X_train, Y_train, Return_train, window, moving, over, NN=NN1 , epoch=10, batch=100):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        model = KerasClassifier(build_fn=NN, epochs=epoch, batch_size=batch, verbose=0, workers = -1, use_multiprocessing = True)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        model.fit(X, Y)
        y_pred[j:j+1] = model.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_sklearn_NN1(X_train, Y_train, Return_train, window, moving, over):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(32),activation = 'relu',max_iter=4000)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        clf.fit(X, Y)
        y_pred[j:j+1] = clf.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_sklearn_NN2_over(X_train, Y_train, Return_train, window, moving, over):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(32,16),activation = 'relu',max_iter=4000)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        clf.fit(X, Y)
        y_pred[j:j+1] = clf.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_sklearn_NN3_over(X_train, Y_train, Return_train, window, moving, over):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(32,16,8),activation = 'relu',max_iter=4000)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        clf.fit(X, Y)
        y_pred[j:j+1] = clf.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_sklearn_NN4_over(X_train, Y_train, Return_train, window, moving, over):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(32,16,8,4),activation = 'relu',max_iter=4000)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        clf.fit(X, Y)
        y_pred[j:j+1] = clf.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])

def Window_sklearn_NN5_over(X_train, Y_train, Return_train, window, moving, over):
    y_pred = copy.deepcopy(Y_train)
    y_pred[:window] = np.nan
    for i, j in enumerate(range(window, len(Y_train))):
        n = i if moving == 1 else 0
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(32,16,8,4,2),activation = 'relu',max_iter=4000)
        X, Y = Oversampling(X_train[n:j], Y_train[n:j]) if over == 1 else Undersampling(X_train[n:j], Y_train[n:j])
        clf.fit(X, Y)
        y_pred[j:j+1] = clf.predict(X_train[j:j+1])
    return Performance(Y_train[912:], y_pred[912:]), Model_return(Return_train[912+1:], y_pred[912:-1])