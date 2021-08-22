import copy
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, fbeta_score, roc_auc_score, precision_score

def Undersampling(X_test, Y_test):
    nm = NearMiss()
    return nm.fit_resample(X_test, Y_test)

def Oversampling(X_test, Y_test):
    smote = SMOTE()
    return smote.fit_resample(X_test, Y_test)

def Performance(Y_test, y_pred):
    results = []
    performance_funcs = [precision_score, recall_score, fbeta_score, roc_auc_score, accuracy_score]
    for func in performance_funcs:
        if func == fbeta_score: results.append(func(Y_test, y_pred,beta = 5))
        else: results.append(func(Y_test, y_pred))
    return results

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

def Plot_return(returns):
    df = returns.T
    df.columns = ['180_moving_window', '365_moving_window', '730_moving_window']
    return df.plot.line()

def Plot_return_baseline(returns, baseline):
    df = returns.T
    df['baseline'] = baseline
    df.columns = ['180_moving_window', '365_moving_window', '730_moving_window', 'baseline']
    return df

test_period = 1096      # Test period start at 1096
def Window_LogReg(X_test, Y_test, Return_test, window, moving, over, penalty='l2'):
    y_pred = copy.deepcopy(Y_test)
    for i, j in enumerate(range(window, len(Y_test))):
        n = i if moving == 1 else 0
        log = LogisticRegression(penalty, solver='lbfgs', max_iter=4000)
        X, Y = Oversampling(X_test[n:j], Y_test[n:j]) if over == 1 else Undersampling(X_test[n:j], Y_test[n:j])
        log.fit(X, Y)
        y_pred[j:j+1] = log.predict(X_test[j:j+1])
    return Performance(Y_test[test_period:], y_pred[test_period:]), Model_return(Return_test[test_period+1:], y_pred[test_period:-1])

def Window_pca(X_test, Y_test, Return_test, window, moving, over, n_components=2, penalty='l2'):
    y_pred = copy.deepcopy(Y_test)
    for i, j in enumerate(range(window, len(Y_test))):
        n = i if moving == 1 else 0
        pca = Pipeline([('pca', PCA(n_components)),('clf', LogisticRegression(penalty, solver='lbfgs', max_iter=4000))])
        X, Y = Oversampling(X_test[n:j], Y_test[n:j]) if over == 1 else Undersampling(X_test[n:j], Y_test[n:j])
        pca.fit(X, Y)
        y_pred[j:j+1] = pca.predict(X_test[j:j+1])
    return Performance(Y_test[test_period:], y_pred[test_period:]), Model_return(Return_test[test_period+1:], y_pred[test_period:-1])

def Window_rf(X_test, Y_test, Return_test, window, moving, over, n=20, d=10):
    n_estimators = n
    max_depth = d
    y_pred = copy.deepcopy(Y_test)
    for i, j in enumerate(range(window, len(Y_test))):
        n = i if moving == 1 else 0
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=123,n_jobs=-1)
        X, Y = Oversampling(X_test[n:j], Y_test[n:j]) if over == 1 else Undersampling(X_test[n:j], Y_test[n:j])
        rf.fit(X, Y)
        y_pred[j:j+1] = rf.predict(X_test[j:j+1])
    return Performance(Y_test[test_period:], y_pred[test_period:]), Model_return(Return_test[test_period+1:], y_pred[test_period:-1])

def Window_KNN(X_test, Y_test, Return_test, window, moving, over, n_neighbors=25):
    y_pred = copy.deepcopy(Y_test)
    for i, j in enumerate(range(window, len(Y_test))):
        n = i if moving == 1 else 0
        KNN = KNeighborsClassifier(n_neighbors,n_jobs=-1)
        X, Y = Oversampling(X_test[n:j], Y_test[n:j]) if over == 1 else Undersampling(X_test[n:j], Y_test[n:j])
        KNN.fit(X, Y)
        y_pred[j:j+1] = KNN.predict(X_test[j:j+1])
    return Performance(Y_test[test_period:], y_pred[test_period:]), Model_return(Return_test[test_period+1:], y_pred[test_period:-1])

dim = 65     # X_test.shape = (1277, 65)
def NN1():
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def NN2():
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def NN3():
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def NN4():
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def NN5():
    model = Sequential()
    model.add(Dense(dim, input_shape=(dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Window_NN(X_test, Y_test, Return_test, window, moving, over, NN=NN1 , epoch=10, batch=100):
    y_pred = copy.deepcopy(Y_test)
    for i, j in enumerate(range(window, len(Y_test))):
        n = i if moving == 1 else 0
        model = KerasClassifier(build_fn=NN, epochs=epoch, batch_size=batch, verbose=0, workers = -1, use_multiprocessing = True)
        X, Y = Oversampling(X_test[n:j], Y_test[n:j]) if over == 1 else Undersampling(X_test[n:j], Y_test[n:j])
        model.fit(X, Y)
        y_pred[j:j+1] = model.predict(X_test[j:j+1])
    return Performance(Y_test[test_period:], y_pred[test_period:]), Model_return(Return_test[test_period+1:], y_pred[test_period:-1])