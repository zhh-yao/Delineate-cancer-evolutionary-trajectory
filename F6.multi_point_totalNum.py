# In[1]: 
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, svm, neighbors, neural_network
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# In[2]: 
## functions
def totalNumber(cycle, data):
    
    l = []
    
    for i in range(cycle):
        l.append(sum(data.iloc[i]))
        
    return l


def dataImport(Pattern, keys, rep, cycle):
    
    data = pd.DataFrame(columns=['cycle_%d'%i for i in range(cycle)])
    
    for i in range(rep):
        mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, i)).item())
        totalNum = totalNumber(cycle, mat)
        
        data = data.append(pd.DataFrame([totalNum], columns=['cycle_%d'%i for i in range(cycle)]))
                
    return data


def labelGet(n=300):
    
    l1 = [0]*n
    l2 = [1]*n
    l3 = [2]*n
    
    return np.array(l1+l2+l3)


def corssValidation(model, data, label, stand=True, n=5, seed=2021):
    
    scores = []
    
    skf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)
    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        if stand == True:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    mean = np.mean(scores)

    return mean


def corssValidation2lgb(para, data, label, n=5, seed=2021):
    
    scores = []
    
    skf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)
    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        
        gbm = lgb.train(para, lgb_train, num_boost_round=20, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=5)
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        y_pred=[list(x).index(max(x)) for x in y_pred]
        score = accuracy_score(y_test, y_pred)
        
        scores.append(score)
    
    mean = np.mean(scores)

    return mean


def multiCycle(data, label, model, cycle):
    '''cycle >= 2'''
    acc = []
    
    for i in range(2, cycle):
        s = corssValidation(model, data.iloc[:,:i], label, stand=True, n=5, seed=2021)
        acc.append(s)
        
    return acc


def multiCycle2lgb(data, label, para, cycle):
    '''cycle >= 2'''
    acc = []
    
    for i in range(2, cycle):
        s = corssValidation2lgb(para, data.iloc[:,:i], label, n=5, seed=2021)
        acc.append(s)
        
    return acc


def plot(data):
    
    n, c = data.shape
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'35'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    t = np.arange(c)
    
    p1, = ax.plot(t, data[0,:], linewidth=10)
    p2, = ax.plot(t, data[1,:], linewidth=10)
    p3, = ax.plot(t, data[2,:], linewidth=10)
    p4, = ax.plot(t, data[3,:], linewidth=10)
    p5, = ax.plot(t, data[4,:], linewidth=10)
    p6, = ax.plot(t, data[5,:], linewidth=10)
    
    
    a=ax.legend([p1,p2,p3], ['LR','SVM','KNN'], loc='best', prop=font2, markerscale = 3, frameon=False)
    ax.legend([p4,p5,p6], ['RF','NN','LGB'], loc="upper center", prop=font2, markerscale = 3, frameon=False)
    plt.gca().add_artist(a)

    plt.ylim([0.4, 0.8])
    
    ax.set_xlabel('Time (cell cycle)', font1)
    ax.set_ylabel('Accuracy', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)
    
    plt.savefig('accuracy_by_num.tif', bbox_inches='tight')
    plt.savefig('accuracy_by_num.eps', bbox_inches='tight')
    plt.show()

# In[3]: 
## predict growth pattern through total number of cells
keys = ['D:0.0,R:0.9,P:0.1',
        'D:0.0,R:1.0,P:0.0',
        'D:0.1,R:0.7,P:0.2',
        'D:0.1,R:0.8,P:0.1',
        'D:0.1,R:0.9,P:0.0',
        'D:0.2,R:0.5,P:0.3',
        'D:0.2,R:0.6,P:0.2',
        'D:0.2,R:0.7,P:0.1',
        'D:0.3,R:0.3,P:0.4',
        'D:0.3,R:0.4,P:0.3',
        'D:0.3,R:0.5,P:0.2',
        'D:0.4,R:0.1,P:0.5',
        'D:0.4,R:0.2,P:0.4',
        'D:0.4,R:0.3,P:0.3',
        'D:0.5,R:0.0,P:0.5',
        'D:0.5,R:0.1,P:0.4']

cycle = 100

dataE = dataImport('Exponential', keys, 300, cycle)
dataL = dataImport('Logistic', keys, 300, cycle)
dataH = dataImport('Hill', keys, 300, cycle)

X = pd.concat([dataE, dataL, dataH], ignore_index=True)
y = labelGet(300)

## logistic regression
lr = LogisticRegression(penalty='l2', C=100, multi_class='ovr', random_state=2021)
accuracy1 = multiCycle(X, y, lr, cycle)
## support vector machine
sv = svm.SVC(decision_function_shape='ovr', random_state=2021)
accuracy2 = multiCycle(X, y, sv, cycle)
## k nearest neighbors
knn = neighbors.KNeighborsClassifier(5)
accuracy3 = multiCycle(X, y, knn, cycle)
## random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
accuracy4 = multiCycle(X, y, rf, cycle)
# neural network
nn = neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6,4), random_state=2021)
accuracy5 = multiCycle(X, y, nn, cycle)
## lightGBM
para = {'task':'train', 'boosting_type':'gbdt', 'objective':'multiclass', 'num_class':3, 'metric':'multi_logloss',
        'num_leaves':10, 'learning_rate':0.05, 'feature_fraction':0.9, 'bagging_fraction':0.8, 'bagging_freq':5}
accuracy6 = multiCycle2lgb(X, y, para, cycle)
##combine six accuracy vectors
accuracy = np.array([accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6])

plot(accuracy)





