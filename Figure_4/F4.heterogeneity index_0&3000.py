# In[1]: 
# packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
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


def dataCombine(dataE, dataL, dataH):
    
    mat = pd.DataFrame(columns=['Pattern', 'Number'])
    
    mat['Pattern'] = ['Aggressive']*len(dataE) + ['Bounded']*len(dataL) + ['Indolent']*len(dataH)
    mat['Number'] = dataE + dataL + dataH
    
    return mat


def main(keys, time, rep):
    
    hilE = indexList('Exponential', keys, time, rep)
    hilL = indexList('Logistic', keys, time, rep)
    hilH = indexList('Hill', keys, time, rep)
    
    p = stats.kruskal(hilE, hilL, hilH)[1]
    f = dataCombine(hilE, hilL, hilH)
    
    return f, p
    

def boxPlot(data, time):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'60'}
    
    fig, ax = plt.subplots(figsize=(13,5))
    sns.boxplot(x=data.columns[0], y=data.columns[1], data=data, width=0.5, palette=['lightcoral','darkgray','cadetblue'], linewidth=1)
    sns.swarmplot(x=data.columns[0], y=data.columns[1], data=data, color='lightsteelblue')
    
    ax.tick_params(axis='x',labelsize=30)
    ax.tick_params(axis='y',labelsize=30)
    
    plt.ylim([-0.1, 2])
    plt.ylim([0.95, 2.05])
    
    #ax.set_xlabel('Pattern', font1)
    #ax.set_ylabel('Heterogeneity index', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)

    plt.savefig('heterogeneity_%d.tif' %time, bbox_inches='tight')
    plt.savefig('heterogeneity_%d.pdf' %time, bbox_inches='tight')


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

rep  = 200
f0, p0 = main(keys, 0, rep)

boxPlot(f0, 0)


rep  = 200
f2200, p = main(keys, 2200, rep)

boxPlot(f2200, 2200)
















