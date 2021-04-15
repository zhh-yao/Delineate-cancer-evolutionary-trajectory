# In[2]: 
# packages
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


# In[2]: 
## ode model
def Exponential(x, t, Para):

    r,a = Para
    
    return np.array(r*x-a*x)


def Logistic(x, t, para):

    r,k,a = para
    
    return np.array(r*x*(1-x/k)-a*x)


def Hill(x, t, para):
    
    r,b,c,a = para
    
    return np.array(r*x**b/(1+c*x**b)-a*x)


def CellNumber(Cycle, model, Para, N0=5000000000):
    
    Times = np.arange(0, Cycle+1)
    CellNum = odeint(model, N0, Times, args = (Para,))
    
    return CellNum

## probabilistic cellular model
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


def plotDynamic(data, allcycle, Pattern):
    
    colors = ['thistle','rosybrown','yellowgreen','lightsalmon','lightskyblue','darkseagreen','cornflowerblue','tomato','violet','plum','peru','khaki','steelblue','orange','wheat','tan']
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'35'}
    time = np.arange(allcycle+1)
    
    fig, ax = plt.subplots(figsize=(13,11))
    ax.stackplot(time, data, colors=colors)
    
    t1 = np.arange(2000+1)
    y = odeint(Exponential, 5000000000, t1, args = (para1,))
    ax.plot(t1, y, linewidth=10, ls='--', color='lightcoral', label='Exponential')
    
    t2 = np.arange(3500+1)
    y = odeint(Logistic, 5000000000, t2, args = (para2,))
    ax.plot(t2, y, linewidth=10, ls='--', color='darkgray', label='Logistic')
    
    t3 = np.arange(4000+1)
    y = odeint(Hill, 5000000000, t3, args = (para3,))
    ax.plot(t3, y, linewidth=10, ls='--', color='cadetblue', label='Hill')
    
    ax.legend(loc='upper right', prop=font2, markerscale = 3, frameon=False)
    ax.set_xlabel('Time (cell cycle)', font1)
    ax.set_ylabel('Number of cells (×10¹¹)', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)
    
    plt.savefig('evolution_%s.tif' %Pattern, bbox_inches='tight')
    plt.savefig('evolution_%s.eps' %Pattern, bbox_inches='tight')
    plt.show()


# In[3]: 
# 1st: calculating different vectors N ———— ode model
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

para1 = [0.01, 0.008]
para2 = [0.01, 500000000000, 0.008]
para3 = [0.01, 1, 0.00000000001, 0.008]

CellNumE = CellNumber(2000, Exponential, para1)[:,0]
CellNumL = CellNumber(3500, Logistic, para2)[:,0]
CellNumH = CellNumber(4000, Hill, para3)[:,0]

Probability  = [[0.1,0.9,0], [0.2,0.7,0.1], [0.3,0.5,0.2], [0.4,0.3,0.3], [0.5,0.1,0.4]] + [[0,1,0], [0.1,0.8,0.1], [0.2,0.6,0.2], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.5,0,0.5]] + [[0,0.9,0.1], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.3,0.3,0.4], [0.4,0.1,0.5]]
# 2nd: plot exponential
ExpMatE = optionalMatrix(Probability, 2000)
coefE = coefCalculate(CellNumE, ExpMatE)
cellMatE = cellMatrix(ExpMatE, coefE)
comCloneE = combineClone(keys, cellMatE, 2000)
dataE = pd.DataFrame(comCloneE, columns=keys)
plotDynamic(dataE.values.T, 2000, 'Exponential')

# 3rd: plot logistic
ExpMatL = optionalMatrix(Probability, 3500)
coefL = coefCalculate(CellNumL, ExpMatL)
cellMatL = cellMatrix(ExpMatL, coefL)
comCloneL = combineClone(keys, cellMatL, 3500)
dataL = pd.DataFrame(comCloneL, columns=keys)
plotDynamic(dataL.values.T, 3500, 'Logistic')

# 4nd: plot hill
ExpMatH = optionalMatrix(Probability, 4000)
coefH = coefCalculate(CellNumH, ExpMatH)
cellMatH = cellMatrix(ExpMatH, coefH)
comCloneH = combineClone(keys, cellMatH, 4000)
dataH = pd.DataFrame(comCloneH, columns=keys)
plotDynamic(dataH.values.T, 4000, 'Hill')














