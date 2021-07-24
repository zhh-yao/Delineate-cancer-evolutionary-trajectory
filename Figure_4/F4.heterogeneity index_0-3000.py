# In[1]: 
# packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[2]: 

def dataImport(Pattern, agent, keys, time):
    
    all_mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, agent), allow_pickle=True).item(), columns=keys)
    
    mat = all_mat.iloc[time]      
         
    return mat
    

def dataProportion(data):
    
    composition = []
        
    for i in data:
        ratio = i/sum(data)
        composition.append(ratio)
        
    return composition


def shannon(composition):
    
    index = 0
    
    for k in composition:
        if k > 0:
            index -= k*math.log(k)
        
    return index


def indexGet(Pattern, agent, keys, time):
    
    data = dataImport(Pattern, agent, keys, time)
    composition = dataProportion(data)
    hi = shannon(composition)
    
    return hi


def indexList(Pattern, keys, time, rep):
    
    hil = []
    
    for agent in range(rep):
        hil.append(indexGet(Pattern, agent, keys, time))
    
    return hil


def mean_confidence_interval(data, confidence=0.95):
    
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def continuousIndex(Pattern, keys, allCycle, rep):
    
    mean, ciu, cil = [], [], []
    
    for time in range(0, allCycle+1, 100):
        
        hil = indexList(Pattern, keys, time, rep)
        m, c1, c2 = mean_confidence_interval(hil, confidence=0.95)
        
        mean.append(m)
        ciu.append(c1)
        cil.append(c2)
        
    return mean, ciu, cil


def fillPlot(dataE, dataL, dataH, allCycle):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    x = np.arange(0, allCycle+1, 100)
    
    ax.fill_between(x, dataE[1], dataE[2], color='lightcoral', alpha=0.3)
    ax.plot(x, dataE[0], linewidth=5, color='lightcoral', label='Exponential')      
    
    ax.fill_between(x, dataL[1], dataL[2], color='darkgray', alpha=0.3)
    ax.plot(x, dataL[0], linewidth=5, color='darkgray', label='Logistic')      
    
    ax.fill_between(x, dataH[1], dataH[2], color='cadetblue', alpha=0.3)
    ax.plot(x, dataH[0], linewidth=5, color='cadetblue', label='Hill')      
    
    ax.tick_params(axis='x',labelsize=30)
    ax.tick_params(axis='y',labelsize=30)
    
    plt.ylim([0.1, 2])
    
    ax.set_xlabel('Time (Cell Cycle)', font1, labelpad=20)
    ax.set_ylabel('Heterogeneity index', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)
    
    ax.legend(loc='best', prop=font1, frameon=False)

    plt.savefig('heterogeneity.tif', bbox_inches='tight')
    plt.savefig('heterogeneity.pdf', bbox_inches='tight')
    

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

allCycle = 3000
rep = 200
mci_E = continuousIndex('Exponential', keys, allCycle, rep)
mci_L = continuousIndex('Logistic', keys, allCycle, rep)
mci_H = continuousIndex('Hill', keys, allCycle, rep)

fillPlot(mci_E, mci_L, mci_H, allCycle)












