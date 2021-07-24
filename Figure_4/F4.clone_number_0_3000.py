# In[1]:

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


# In[2]: 

def dataImport(Pattern, agent, keys):
    
    mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, agent), allow_pickle=True).item(), columns=keys)
                
    return mat


def mat2bool(mat):
    
    x = pd.DataFrame(columns=mat.columns)
    
    for i in range(0, len(mat)+1, 100):
        temp = []
        matR = mat.loc[i]
        for j in matR:
            if j >= 1:
                temp.append(1)
            else:
                temp.append(0)
        
        x.loc[i//100] = temp
        
    return x


def num_subclone(mat):
    
    num = []
    for i in range(len(mat)):
        num.append(len(np.where(mat.loc[i]>0)[0]))
        
    return num


def all_num_subclone(Pattern, rep, keys):
    
    all_num = []
    
    for agent in range(rep):
        
        mat = dataImport(Pattern, agent, keys)
        matbool = mat2bool(mat)
        num = num_subclone(matbool)
        
        all_num.append(num)
      
    return all_num
 

def mean_confidence_interval(data, confidence=0.95):
    
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def mci(all_num):
    
    mean, ciu, cil = [], [], []
    rep = len(all_num)
    cycle = len(all_num[0])
    
    for i in range(cycle):
        temp = []
        for j in range(rep):
            temp.append(all_num[j][i])
        m, c1, c2 = mean_confidence_interval(temp, confidence=0.95)
        
        mean.append(m)
        ciu.append(c1)
        cil.append(c2)
        
    return mean, ciu, cil


def fillPlot(mE, mL, mH, allCycle):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'normal', 'size':'40'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    x = np.arange(0, allCycle+1, 100)
    
    ax.fill_between(x, mE[1], mE[2], color='lightcoral', alpha=0.3)
    ax.plot(x, mE[0], linewidth=5, color='lightcoral', label='Exponential')      
    
    ax.fill_between(x, mL[1], mL[2], color='darkgray', alpha=0.3)
    ax.plot(x, mL[0], linewidth=5, color='darkgray', label='Logistic')      
    
    ax.fill_between(x, mH[1], mH[2], color='cadetblue', alpha=0.3)
    ax.plot(x, mH[0], linewidth=5, color='cadetblue', label='Hill')      
    
    ax.tick_params(axis='x',labelsize=30)
    ax.tick_params(axis='y',labelsize=30)
    
    ax.set_xlabel('Time (Cell Cycle)', font1, labelpad=20)
    ax.set_ylabel('Number of subclones', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)
    
    ax.legend(loc='best', prop=font2, frameon=False)

    plt.savefig('cloneNum.tif', bbox_inches='tight')
    plt.savefig('cloneNum.eps', bbox_inches='tight')
    

# In[3]: 

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


all_numE = all_num_subclone('Exponential', 200, keys)
all_numL = all_num_subclone('Logistic', 200, keys)
all_numH = all_num_subclone('Hill', 200, keys)


mci_E = mci(all_numE)
mci_L = mci(all_numL)
mci_H = mci(all_numH)


fillPlot(mci_E, mci_L, mci_H, 3000)




