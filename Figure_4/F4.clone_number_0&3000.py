# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# In[2]: 

def dataImport(Pattern, agent, keys):
    
    mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, agent), allow_pickle=True).item(), columns=keys)
                
    return mat


def num_cycle(mat, cycle):
    
    temp = []
    matR = mat.loc[cycle]
    
    for i in matR:
        if i >= 1:
            temp.append(1)
        else:
            temp.append(0)
        
    return np.count_nonzero(temp)


def all_num_cycle(Pattern, rep, keys, cycle):
    
    num = []
    
    for agent in range(rep):
        mat = dataImport(Pattern, agent, keys)
        num.append(num_cycle(mat, cycle))
        
    return num


def dataCombine(dataE, dataL, dataH):
    
    mat = pd.DataFrame(columns=['Pattern', 'Number'])
    
    mat['Pattern'] = ['Exponential']*len(dataE) + ['Logistic']*len(dataL) + ['Hill']*len(dataH)
    mat['Number'] = dataE + dataL + dataH
    
    return mat


def boxPlot(data, time):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'60'}
    
    fig, ax = plt.subplots(figsize=(13,5))
    
    sns.boxplot(x=data.columns[0], y=data.columns[1], data=data, width=0.5, palette=['lightcoral','darkgray','cadetblue'], linewidth=1)
    sns.swarmplot(x=data.columns[0], y=data.columns[1], data=data, color='lightsteelblue')
    
    ax.tick_params(axis='x',labelsize=50)
    ax.tick_params(axis='y',labelsize=50)
    
    plt.ylim([0.5, 14.5])
    plt.ylim([5.2, 17.2])
    
    #ax.set_xlabel('Pattern', font1, labelpad=20)
    #ax.set_ylabel('Number of subclones', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)
    
    
    plt.savefig('cloneNum_%d.tif' %time, bbox_inches='tight')
    plt.savefig('cloneNum_%d.pdf' %time, bbox_inches='tight')


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

# cell cycle 0
numE_0 = all_num_cycle('Exponential', 200, keys, 0)
numL_0 = all_num_cycle('Logistic', 200, keys, 0)
numH_0 = all_num_cycle('Hill', 200, keys, 0)

data_0 = dataCombine(numE_0, numL_0, numH_0)

p_0 = stats.kruskal(numE_0, numL_0, numH_0)[1]

boxPlot(data_0, 0) #boxPlot(data_0, p_0)

# cell cycle 2000
numE_2200 = all_num_cycle('Exponential', 200, keys, 2200)
numL_2200 = all_num_cycle('Logistic', 200, keys, 2200)
numH_2200 = all_num_cycle('Hill', 200, keys, 2200)

data_2200 = dataCombine(numE_2200, numL_2200, numH_2200)

p_2200 = stats.kruskal(numE_2200, numL_2200, numH_2200)[1]

boxPlot(data_2200, 2200)

# cell cycle 3000
numE_3000 = all_num_cycle('Exponential', 200, keys, 3000)
numL_3000 = all_num_cycle('Logistic', 200, keys, 3000)
numH_3000 = all_num_cycle('Hill', 200, keys, 3000)

data_3000 = dataCombine(numE_3000, numL_3000, numH_3000)

p_3000 = stats.kruskal(numE_3000, numL_3000, numH_3000)[1]

boxPlot(data_3000, 3000) 




