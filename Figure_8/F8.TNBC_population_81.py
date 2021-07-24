# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# In[2]:
# calculate possible grawth
def number_estimation(diameter):
    
    number = (diameter*10/2)**3*10**6
    
    return number


def Exponential(x, t, para):

    r,a = para
    
    return np.array(r*x-a*x)


def Logistic(x, t, para):

    r,k,a = para
    
    return np.array(r*x*(1-x/k)-a*x)


def Hill(x, t, para):
    
    r,b,c,a = para
    
    return np.array(r*x**b/(1+c*x**b)-a*x)


def CellNumber(Cycle, model, Para, N0):
    
    Times = np.arange(0, Cycle+1)
    CellNum = odeint(model, N0, Times, args = (Para,))
    
    return CellNum


def hypothesis_Plot(Patient, growth1, growth2, growth3):
    
    fig, ax = plt.subplots(figsize=(13,11))
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)        
    
    time1 = np.arange(0, len(growth1))
    time2 = np.arange(0, len(growth2))
    time3 = np.arange(0, len(growth3))
    
    ax.plot(time1, growth1, color='lightcoral', linewidth=10, ls='--', label='Aggressive', zorder=1)
    ax.plot(time2, growth2, color='darkgray', linewidth=10, ls='--', label='Bounded', zorder=1)
    ax.plot(time3, growth3, color='cadetblue', linewidth=10, ls='--', label='Indolent', zorder=1)
    
    ax.scatter(0, growth1[0], c='k', marker='x', linewidth=50, zorder=2)
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    ax.legend(loc='upper left', prop=font1, frameon=False)
    
    #plt.xlim([-50, time[-1]+50])
    #plt.ylim([0, 1000])
    #plt.ylim([50000000000, 1000000000000])
    
    ax.set_xlabel('Time (days)', font1)
    ax.set_ylabel('Number of cells (×10¹º)', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, width=2, labelsize=40)
    
    plt.savefig('Pt%s.tif' %Patient, bbox_inches='tight')
    plt.savefig('Pt%s.eps' %Patient, bbox_inches='tight')
    plt.show()


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
Probability  = [[0.1,0.9,0], [0.2,0.7,0.1], [0.3,0.5,0.2], [0.4,0.3,0.3], [0.5,0.1,0.4]] + [[0,1,0], [0.1,0.8,0.1], [0.2,0.6,0.2], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.5,0,0.5]] + [[0,0.9,0.1], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.3,0.3,0.4], [0.4,0.1,0.5]]

number_58 = number_estimation(1.5)
t = 2000
# calculate possible grawth
Para1 = [0.01, 0.008]
Para2 = [0.01, 40000000000, 0.008]
Para3 = [0.01, 1, 0.0000000001, 0.008]

growth1 = CellNumber(t, Exponential, Para1, number_58)[:,0]
growth2 = CellNumber(t, Logistic, Para2, number_58)[:,0]
growth3 = CellNumber(t, Hill, Para3, number_58)[:,0]
hypothesis_Plot('058', growth1, growth2, growth3)


# In[3]:
# generate training data
def randomRatio(Probability, cellNum=1000):
    
    multiNum = np.random.multinomial(n=cellNum, pvals=Probability, size=1)
    Num = multiNum[0][1]*1 + multiNum[0][2]*2
    ratio = Num/cellNum
    
    return ratio


def Expectation(Probability, initialExp, cycle):
    
    ratio = randomRatio(Probability)
    nextExp = [ratio*initialExp[-1]]
    
    for i in range(1, cycle):
        ratio = randomRatio(Probability)
        nextExp.append(ratio*nextExp[i-1])
    
    return initialExp + nextExp


def ExpectationMatrix(Probability, evolveTime, allcycle):
    
    ExpMat = pd.DataFrame()
    
    for Prob1 in Probability:
        Exp1 = Expectation(Prob1, [1], cycle=evolveTime)
        for Prob2 in Probability:
            Exp2 = Expectation(Prob2, Exp1, cycle= allcycle - evolveTime)
            ExpMat['D:%.1f,R:%.1f,P:%.1f+D:%.1f,R:%.1f,P:%.1f+%d'%(Prob1[0],Prob1[1],Prob1[2],Prob2[0],Prob2[1],Prob2[2],evolveTime)] = Exp2
    
    return ExpMat


def optionalMatrix(Probability, allcycle):
    
    opMat = pd.DataFrame()
    
    for i in np.arange(100, allcycle, 100):
        mat = ExpectationMatrix(Probability, i, allcycle)
        opMat = pd.concat([opMat, mat], axis=1)
    
    return opMat


def coefCalculate(CellNum, ExpMat):
    
    clf = Lasso(alpha=1, positive=True, fit_intercept=False, tol=0.1)
    clf.fit(ExpMat, CellNum)
    coef = clf.coef_
    
    return coef


def cellMatrix(ExpMat, coef):
    
    mat = pd.DataFrame()
    
    ind = np.where(coef>0)[0]
    for i in ind:
        mat[ExpMat.columns[i]] = coef[i]*ExpMat[ExpMat.columns[i]]
    
    return mat


def numberExtend(data, colName):
    
    time = int(colName[36:])
    Num1 = list(data[:(time+1)])
    Num2 = list(data[(time+1):])
    
    extentNum1 = Num1 + [0]*(len(data)-len(Num1))
    extentNum2 = [0]*(len(data)-len(Num2)) + Num2
    
    beforeDict = {colName[:17]: extentNum1}
    afterDict = {colName[18:35]: extentNum2}
    
    return beforeDict, afterDict


def combineClone(keys, cellMat, allcycle):
    
    comClone = dict.fromkeys(keys, [0]*(allcycle+1))
    
    for col in cellMat.columns:
        data = cellMat[col]
        beforeDict, afterDict = numberExtend(data, col)
        
        comClone[col[:17]] = np.sum([comClone[col[:17]], beforeDict[col[:17]]],0)
        comClone[col[18:35]] = np.sum([comClone[col[18:35]], afterDict[col[18:35]]],0)
    
    return comClone


def agentMat(Patient, CellNum1, CellNum2, CellNum3, Probability, keys, allcycle, rep):
    
    for i in range(0, rep):

        ExpMat = optionalMatrix(Probability, allcycle)
        
        coef1 = coefCalculate(CellNum1, ExpMat)
        coef2 = coefCalculate(CellNum2, ExpMat)
        coef3 = coefCalculate(CellNum3, ExpMat)
        cellMat1 = cellMatrix(ExpMat, coef1)
        cellMat2 = cellMatrix(ExpMat, coef2)
        cellMat3 = cellMatrix(ExpMat, coef3)
        comClone1 = combineClone(keys, cellMat1, allcycle)
        comClone2 = combineClone(keys, cellMat2, allcycle)
        comClone3 = combineClone(keys, cellMat3, allcycle)
        
        np.save('Pt%s_%d_%d.npy'%(Patient, 1, i), comClone1)
        np.save('Pt%s_%d_%d.npy'%(Patient, 2, i), comClone2)
        np.save('Pt%s_%d_%d.npy'%(Patient, 3, i), comClone3)


# generate training data
        
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
Probability  = [[0.1,0.9,0], [0.2,0.7,0.1], [0.3,0.5,0.2], [0.4,0.3,0.3], [0.5,0.1,0.4]] + [[0,1,0], [0.1,0.8,0.1], [0.2,0.6,0.2], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.5,0,0.5]] + [[0,0.9,0.1], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.3,0.3,0.4], [0.4,0.1,0.5]]

agentMat('058', growth1, growth2, growth3, Probability, keys, t, rep=100)


# In[4]:

# train classifier: decision tree
def dataImport(Patient, Pattern, keys, rep, timeP):
    
    newKeys = []
    for i in timeP:
        newKeys = newKeys + [k+'.'+str(i) for k in ['000901','001000','010702','010801','010900','020503','020602','020701',
        '030304','030403','030502','040105','040204','040303','050005','050104']]
    
    data = pd.DataFrame(columns=newKeys)
    
    for i in range(rep):
        mat = pd.DataFrame(np.load('Pt%s_%d_%d.npy'%(Patient, Pattern, i)).item())
        
        temp = pd.Series()
        for j in timeP:
            temp = temp.append(mat[keys].iloc[j])
        
        data = data.append(pd.DataFrame([temp.values],columns=newKeys), ignore_index=True)
                
    return data


def dataMerge(Patient, keys, rep, timeP):
    
    data1 = dataImport(Patient, 1, keys, rep, timeP)
    data2 = dataImport(Patient, 2, keys, rep, timeP)
    data3 = dataImport(Patient, 3, keys, rep, timeP)

    data = pd.concat([data1, data2, data3])
    data = data.set_index(np.arange(3*rep))
    
    return data


def dataProportion(data):
    
    n = len(data)
    
    data[data < 1] = 0
    
    for i in range(n):
        data.loc[i] = data.loc[i]/sum(data.loc[i])
        
    return data

'''
def labelGet(n=100):
    
    l1 = [1]*n
    l2 = [2]*2*n
    
    return np.array(l1+l2)

'''
def labelGet(n=100):
    
    l1 = [1]*2*n
    l2 = [2]*n
    
    return np.array(l1+l2)
'''
def labelGet(n=100):
    
    l1 = [1]*n
    l2 = [2]*n
    l3 = [3]*n
    
    return np.array(l1+l2+l3)
'''

def corssValidation(model, data, label, n=5, seed=2021):
    
    scores = []
    
    skf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)
    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    mean = np.mean(scores)

    return mean, scores


def feat_importance(model, data, label, n=5, seed=2021):
    
    importances = 0

    skf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)
    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        model.fit(X_train, y_train)
        results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=2)
        importances += results.importances_mean
        
    return importances/n


def accumulated_importance(importance):
    
    accu_importance = []
    
    for i in range(len(importance)):
        
        temp = sum(importance[:i+1])/sum(importance)
        accu_importance.append(temp)
    
    return accu_importance


def plotImportance(Patient, feature, importance):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'40'}
    
    fig = plt.figure(figsize=(15,10))
    
    ax1 = fig.add_subplot(111)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    
    sorted_idx = importance.argsort()[::-1]
    sorted_feature = [feature[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    ax1.bar(sorted_feature, sorted_importance, color='cornflowerblue', label='Feature importance') 
    ax1.axhline(y=0, lw=2, color='k')
    ax1.tick_params(length=10, width=2, labelsize=40)
    plt.xticks(rotation=70)
    plt.ylim([-0.027, 0.5])
    plt.legend(loc=(0.2, 0.6), prop=font1, frameon=False)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
        
    ax2 = ax1.twinx()
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    
    accu_importance = accumulated_importance(sorted_importance)
    ax2.plot(sorted_feature, accu_importance, lw=10, ls='-.', label='Accumulated importance', color='palevioletred')
    ax2.tick_params(length=10, width=2, labelsize=40)
    plt.ylim([-0.06, 1.1])
    ax1.set_xlabel('Subclones(D=Death, R=Resting, P=Proliferation)', font1)
    plt.legend(loc=(0.2, 0.5), prop=font1, frameon=False)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    
    plt.savefig('Pt%s_feature_importance.tif' %Patient, bbox_inches='tight')
    plt.savefig('Pt%s_feature_importance.eps' %Patient, bbox_inches='tight')
    plt.show()


# train classifier: decision tree
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
Probability  = [[0.1,0.9,0], [0.2,0.7,0.1], [0.3,0.5,0.2], [0.4,0.3,0.3], [0.5,0.1,0.4]] + [[0,1,0], [0.1,0.8,0.1], [0.2,0.6,0.2], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.5,0,0.5]] + [[0,0.9,0.1], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.3,0.3,0.4], [0.4,0.1,0.5]]
X = dataMerge('081', keys, 100, [0])
X = dataProportion(X)
y = labelGet(100)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
m_acc, acc = corssValidation(rf, X, y, n=5, seed=2021)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
rf.fit(X, y)
rf.score(X, y)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
importance = feat_importance(rf, X, y, n=5, seed=2021)
plotImportance('081', keys, importance)

X_part = X[['001000.0', '050005.0']]
X_part = X[['001000.0', '050005.0', '010801.0']]
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
rf.fit(X_part, y)
rf.score(X_part, y)

rf.predict([(p81_1, p81_2)])
rf.predict([(p84_1, p84_2)])



def dataMerge(Patient, keys, rep, timeP):
    
    data1 = dataImport(Patient, 1, keys, rep, timeP)
    data2 = dataImport(Patient, 2, keys, rep, timeP)

    data = pd.concat([data1, data2])
    data = data.set_index(np.arange(2*rep))
    
    return data


def labelGet(n=100):
    
    l1 = [1]*n
    l2 = [2]*n
    
    return np.array(l1+l2)

Probability  = [[0.1,0.9,0], [0.2,0.7,0.1], [0.3,0.5,0.2], [0.4,0.3,0.3], [0.5,0.1,0.4]] + [[0,1,0], [0.1,0.8,0.1], [0.2,0.6,0.2], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.5,0,0.5]] + [[0,0.9,0.1], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.3,0.3,0.4], [0.4,0.1,0.5]]
X = dataMerge('081', keys, 100, [0])
X = dataProportion(X)
y = labelGet(100)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
m_acc, acc = corssValidation(rf, X, y, n=5, seed=2021)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
rf.fit(X, y)
rf.score(X, y)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
importance = feat_importance(rf, X, y, n=5, seed=2021)
plotImportance('081', keys, importance)


X_part = X[['001000.0', '050005.0', '010801.0']]
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2021)
rf.fit(X_part, y)
rf.score(X_part, y)

rf.predict([(p81_1, p81_2, p81_3)])
rf.predict([(p84_1, p84_2, p84_3)])










