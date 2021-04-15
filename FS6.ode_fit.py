# In[1]:

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

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


def fitCellNumber(ExpMat, CellNum):
    
    clf = Lasso(alpha=1, positive=True, fit_intercept=False, tol=0.1)
    clf.fit(ExpMat, CellNum)
    score = clf.score(ExpMat, CellNum)
    
    coef = clf.coef_
    newCellNum = np.dot(ExpMat, coef)
    
    return score, newCellNum


def fitPlot(para1, para2, para3, newCellNumE, newCellNumL, newCellNumH):
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    tE = np.arange(0, 2001, 1)
    tL = np.arange(0, 3501, 1)
    tH = np.arange(0, 4001, 1)
    
    y = odeint(Exponential, 5000000000, tE, args = (para1,))
    ax.plot(tE, y, linewidth=1, color='lightcoral', label='Exponential')    
    y = odeint(Logistic, 5000000000, tL, args = (para2,))
    ax.plot(tL, y, linewidth=1, color='darkgray', label='Logistic')    
    y = odeint(Hill, 5000000000, tH, args = (para3,))
    ax.plot(tH, y, linewidth=1, color='cadetblue', label='Hill')    

    for i in range(300):
        random.seed(i)
        paraE = [np.random.normal(para1[0], 0.5*para1[0]**2), para1[1]]
        y = odeint(Exponential, 5000000000, tE, args = (paraE,))
        ax.plot(tE, y, linewidth=1, color='lightcoral', zorder=1)
    
        paraL = [np.random.normal(para2[0], 0.5*para2[0]**2), para2[1], para2[2]]
        y = odeint(Logistic, 5000000000, tL, args = (paraL,))
        ax.plot(tL, y, linewidth=1, color='darkgray', zorder=1)

        paraH = [np.random.normal(para3[0], 0.5*para3[0]**2), para3[1], para3[2], para3[3]]
        y = odeint(Hill, 5000000000, tH, args = (paraH,))
        ax.plot(tH, y, linewidth=1, color='cadetblue', zorder=1)
    
    ax.scatter(tE, newCellNumE, s=2, c='gold', zorder=2)
    ax.scatter(tL, newCellNumL, s=2, c='gold', zorder=2)
    ax.scatter(tH, newCellNumH, s=2, c='gold', zorder=2, label='Fitted curve')
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'45'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'30'}
    ax.set_xlabel('Time (Cell Cycle)', font1)
    ax.set_ylabel('Number of cells (×10¹¹)', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)
    
    l = ax.legend(markerscale=10, loc='best', prop=font2, frameon=False)
    for i in l.legendHandles:
        i.set_linewidth(10)
    
    plt.savefig('fitCurve.tif', bbox_inches='tight')
    plt.savefig('fitCurve.eps', bbox_inches='tight')
    plt.show()
    

# In[3]:
    
para1 = [0.01, 0.008]                         #exponential
para2 = [0.01, 500000000000, 0.008]           #logistic
para3 = [0.01, 1, 0.00000000001, 0.008]     #hill

# 1st: calculating different vectors N ———— ode model
CellNumE = CellNumber(2000, Exponential, para1)[:,0]
CellNumL = CellNumber(3500, Logistic, para2)[:,0]
CellNumH = CellNumber(4000, Hill, para3)[:,0]

# 2nd: calculating average fitting score
scoreES, scoreLS, scoreHS = 0, 0, 0
newCellNumES, newCellNumLS, newCellNumHS = 0, 0, 0

Probability  = [[0.1,0.9,0], [0.2,0.7,0.1], [0.3,0.5,0.2], [0.4,0.3,0.3], [0.5,0.1,0.4]] + [[0,1,0], [0.1,0.8,0.1], [0.2,0.6,0.2], [0.3,0.4,0.3], [0.4,0.2,0.4], [0.5,0,0.5]] + [[0,0.9,0.1], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.3,0.3,0.4], [0.4,0.1,0.5]]
for i in range(10):
    ExpMatE = optionalMatrix(Probability, 2000)
    scoreE, newCellNumE = fitCellNumber(ExpMatE, CellNumE)
    scoreES += scoreE
    newCellNumES += newCellNumE
    
    ExpMatL = optionalMatrix(Probability, 3500)
    scoreL, newCellNumL = fitCellNumber(ExpMatL, CellNumL)
    scoreLS += scoreL
    newCellNumLS += newCellNumL    
    
    ExpMatH = optionalMatrix(Probability, 4000)
    scoreH, newCellNumH = fitCellNumber(ExpMatH, CellNumH)
    scoreHS += scoreH
    newCellNumHS += newCellNumH
# 3th: mean score
scoreE = scoreES/10
scoreL = scoreLS/10
scoreH = scoreHS/10
# 4th: plot
fitPlot(para1, para2, para3, newCellNumES/10, newCellNumLS/10, newCellNumHS/10)
















