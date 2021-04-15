# In[1]: 
# packages
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm, neighbors, neural_network
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# In[2]: 
## ode model
def dataImport(Pattern, keys, rep, timeP):
    
    newKeys = []
    for i in timeP:
        newKeys = newKeys + [k+'.'+str(i) for k in ['000901','001000','010702','010801','010900','020503','020602','020701',
        '030304','030403','030502','040105','040204','040303','050005','050104']]
    
    data = pd.DataFrame(columns=newKeys)
    
    for i in range(rep):
        mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, i)).item())
        
        temp = pd.Series()
        for j in timeP:
            temp = temp.append(mat[keys].iloc[j])
        
        data = data.append(pd.DataFrame([temp.values],columns=newKeys), ignore_index=True)
                
    return data


def dataMerge(keys, rep, timeP):
    
    dataE = dataImport('Exponential', keys, rep, timeP)
    dataL = dataImport('Logistic', keys, rep, timeP)
    dataH = dataImport('Hill', keys, rep, timeP)

    data = pd.concat([dataE, dataL, dataH])
    data = data.set_index(np.arange(3*rep))
    
    return data


def labelGet(n=100):
    
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

    acc = []
    
    for i in range(cycle):
        s = corssValidation(model, data[data.columns[:16*(i+1)]], label, stand=True, n=5, seed=2021)
        acc.append(s)
        
    return acc


def multiCycle2lgb(data, label, para, cycle):

    acc = []
    
    for i in range(cycle):
        s = corssValidation2lgb(para, data[data.columns[:16*(i+1)]], label, n=5, seed=2021)
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

    plt.ylim([0.5, 0.9])
    
    ax.set_xlabel('Time (cell cycle)', font1)
    ax.set_ylabel('Accuracy', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)
    
    plt.savefig('accuracy_by_subclone.tif', bbox_inches='tight')
    plt.savefig('accuracy_by_subclone.eps', bbox_inches='tight')
    plt.show()


'''
def featImportance():
    
    importances = rf.feature_importances_

    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(13,11))
    fig = plt.figure(figsize=(10,6))
    plt.bar(range(len(indices)), importances[indices], color='darkcyan',  align="center")
    plt.step(range(len(indices)), np.cumsum(importances[indices]), color='darkcyan', where='mid', label='Cumulative importance')
    plt.xticks(range(len(indices)), indices, fontsize=15)
    plt.xlim([-1, len(indices)])
    plt.legend(loc='best', fontsize=20)
    plt.savefig('importance.tif', bbox_inches='tight')
    plt.show()
'''

# In[3]: 
## accuracy of different classifier
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

X = dataMerge(keys, 300, np.arange(cycle))
y = labelGet(300)

# six candidate models
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










