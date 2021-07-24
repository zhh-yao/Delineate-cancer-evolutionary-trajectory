# In[1]: 
# packages
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm, neighbors, neural_network
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
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


def labelGet(n=200):
    
    l1 = [0]*n
    l2 = [1]*n
    l3 = [2]*n
    
    return np.array(l1+l2+l3)


def corssValidation(model, data, label, stand=True, n=5, seed=2021):
    
    start_t = time.time()
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
    end_t = time.time()
    dur = end_t-start_t

    return mean, dur, scores


def corssValidation2lgb(para, data, label, n=5, seed=2021):
    
    start_t = time.time()
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
    end_t = time.time()
    dur = end_t-start_t

    return mean, dur, scores


def Confusion(y_test, y_predict, model):
    
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}

    fig, ax = plt.subplots(figsize=(13,11))
    
    p = pd.DataFrame(confusion_matrix(y_test, y_predict, normalize='true'), index=['Aggressive','Bounded','Indolent'],columns=['Aggressive','Bounded','Indolent'])
    sns.heatmap(p, cmap=plt.cm.Blues, annot=True, annot_kws={'size':40}, square=True, cbar=0)
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=-0)
    ax.set_xlabel('Predicted label', font1, labelpad=20)
    ax.set_ylabel('True label', font1, labelpad=20)
    ax.tick_params(length=0, labelsize=30, pad=20)
    
    plt.savefig('con_matrix_%s.tif'% model, bbox_inches='tight')
    plt.savefig('con_matrix_%s.pdf'% model, bbox_inches='tight')


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

X = dataMerge(keys, 200, [0])
y = labelGet(200)

# six candidate models
## logistic regression
lr = LogisticRegression(penalty='l2', C=100, multi_class='ovr', random_state=2021) 
ss1, t1, s1 = corssValidation(lr, X, y, stand=True, n=5, seed=2021)
## support vector machine
sv = svm.SVC(decision_function_shape='ovr', random_state=2021)
ss2, t2, s2 = corssValidation(sv, X, y, stand=True, n=5, seed=2021)
## k nearest neighbors
knn = neighbors.KNeighborsClassifier(5)
ss3, t3, s3 = corssValidation(knn, X, y, stand=True, n=5, seed=2021)
## random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
ss4, t4, s4 = corssValidation(rf, X, y, stand=False, n=5, seed=2021)
# neural network
nn = neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6,4), random_state=2021)
ss5, t5, s5 = corssValidation(nn, X, y, stand=True, n=5, seed=2021)
## lightGBM
para = {'task':'train', 'boosting_type':'gbdt', 'objective':'multiclass', 'num_class':3, 'metric':'multi_logloss',
        'num_leaves':10, 'learning_rate':0.05, 'feature_fraction':0.9, 'bagging_fraction':0.8, 'bagging_freq':5}
ss6, t6, s6 = corssValidation2lgb(para, X, y, n=5, seed=2021)


# In[4]: 
## draw confusion matrix with different classifier
### scaled data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

## logistic regression
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
Confusion(y_test, y_predict, 'lr')
## support vector machine
sv.fit(X_train, y_train)
y_predict = sv.predict(X_test)
Confusion(y_test, y_predict, 'sv')
## k nearest neighbors
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
Confusion(y_test, y_predict, 'knn')
# neural network
nn.fit(X_train, y_train)
y_predict = nn.predict(X_test)
Confusion(y_test, y_predict, 'nn')
### non-scaled data
X_train, X_test, y_train, y_test = train_test_split(X[X.columns[:16]], y, random_state=2021)
## random forest
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)
Confusion(y_test, y_predict, 'rf')
## lightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
gbm = lgb.train(para, lgb_train, num_boost_round=20, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=5)
y_predict = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_predict = [list(x).index(max(x)) for x in y_predict]
Confusion(y_test, y_predict, 'lgb')












