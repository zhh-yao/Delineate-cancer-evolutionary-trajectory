# In[1]:

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import Lasso
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# In[2]:

def dataImport(Patient):
    
    temp = pd.read_csv('CLL_Patients\WBC_counts\%d.txt' %Patient, skiprows=1, header=None, names = ['Year','WBC'])
    
    timeL = [math.ceil(temp['Year'][0]*365)]
    
    for i in range(1, len(temp)):
        t = math.ceil(temp.iloc[i,0]*365)
        timeL.append(t)
        
    data = pd.DataFrame({'Time': timeL, 'WBC': temp['WBC']*1e9})
            
    return data


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


def cloneType(ExpMat, coef):
    
    clone = set()
    
    path = ExpMat.columns[np.where(coef>0)]
    for i in range(len(path)):
        clone.add(path[i][:17])
        clone.add(path[i][18:35])
    
    return clone


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
        
        comClone[col[:17]] = np.sum([comClone[col[:17]], beforeDict[col[:17]]], 0)
        comClone[col[18:35]] = np.sum([comClone[col[18:35]], afterDict[col[18:35]]], 0)
    
    return comClone


def interData(data, kind):
    '''
    kind can be linear, quadratic, cubic and etc.
    '''
    f = interpolate.interp1d(data['Time'], data['WBC'], kind=kind)
    
    xnew = np.arange(data['Time'][0], data['Time'][len(data)-1]+1, 1)
    ynew = f(xnew)
    
    return xnew, ynew


def evolutionDetail(Patient, keys, Probability, xnew, ynew, rep):
    
    for i in range(0, rep):
        ExpMat = optionalMatrix(Probability, xnew[-1]-xnew[0])
        coef = coefCalculate(ynew, ExpMat)
        cellMat = cellMatrix(ExpMat, coef)
        comClone = combineClone(keys, cellMat, xnew[-1]-xnew[0])
        np.save('evolution_%d_%d.npy' %(Patient,i), comClone)



# In[3]:
# Calculating evolution details of Patient 1-21
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

Probability = [[0.1,0.9,0], [0.2,0.7,0.1], [0.3,0.5,0.2], [0.4,0.3,0.3], [0.5,0.1,0.4]] + [[0,1,0], [0.1,0.8,0.1], [0.2,0.6,0.2], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.5,0,0.5]] + [[0,0.9,0.1], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.3,0.3,0.4], [0.4,0.1,0.5]]


for Patient in range(1,22,1):
    
    data = dataImport(Patient)
    xnew, ynew = interData(data, 'linear')
    
    evolutionDetail(Patient, keys, Probability, xnew, ynew, 40)


# In[4]:

def dataReImport(Patient, i, keys):
    
    temp = pd.read_csv('CLL_Patients\WBC_counts\%d.txt' %Patient, skiprows=1, header=None, names = ['Year','WBC'])
    timeI = math.ceil(temp['Year'][0]*365)
    
    mat = pd.DataFrame(np.load('CLL_Patients\subclone_composition\evolution_%d_%d.npy' %(Patient, i), allow_pickle=True).item(), columns=keys)
    mat.index = np.arange(timeI, timeI+len(mat))
    
    return mat


def cloneNumber(data, keys, timeP):
    
    num = 0
    
    for i in range(len(keys)):
        if data.iloc[timeP, i] >= 1:
            num += 1
            
    return num


def numberList(Patient, rep, keys, timeP):
    
    number = []
    
    for i in range(rep):
        mat = dataReImport(Patient, i, keys)
        number.append(cloneNumber(mat, keys, timeP))
        
    return number


def dataCombine(keys, timeP, rep):
    
    PList = []
    NList = []
    
    mat = pd.DataFrame(columns=['Patient', 'Number'])
    
    for i in [16,4,19,18,11, 7,17,10,8,12,15, 20,2,9,3,1,6,5,14,13,21]:
        
        PList = PList + [i]*rep
        NList = NList + numberList(i, rep, keys, timeP)
    
    mat['Patient'] = PList
    mat['Number'] = NList
    
    return mat


def boxPlot(data):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'normal', 'size':'30'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    sns.boxplot(x=data.columns[0], y=data.columns[1], data=data, order=[16,4,19,18,11, 7,17,10,8,12,15, 20,2,9,3,1,6,5,14,13,21])
    
    plt.vlines(4.5, 1.8, 15, colors='k', linestyles='--')
    plt.vlines(10.5, 1.8, 15, colors='k', linestyles='--')
    
    plt.hlines(sum(barData.iloc[0:50, 1])/50, -1, 4.5, colors='red', linestyle='--', lw=4)
    plt.hlines(sum(barData.iloc[50:110, 1])/60, 4.5, 10.5, colors='red', linestyle='--', lw=4)
    plt.hlines(sum(barData.iloc[110:210, 1])/100, 10.5, 21, colors='red', linestyle='--', lw=4)
    
    plt.text(0.5, 2.5, 'Logistic', fontdict=font2)    
    plt.text(4.8, 2.5, 'Indeterminate', fontdict=font2)    
    plt.text(13, 2.5, 'Exponential', fontdict=font2)    
        
    ax.set_xlabel('Patient ID', font1, labelpad=20)
    ax.set_ylabel('Number of subclones', font1, labelpad=20)
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=25, pad=20)

    plt.savefig('subclone_num.tif', bbox_inches='tight')
    plt.savefig('subclone_num.eps', bbox_inches='tight')


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

barData = dataCombine(keys, 0, 40)
boxPlot(barData)












