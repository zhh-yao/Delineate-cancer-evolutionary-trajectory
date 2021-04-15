# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# In[2]: 
def dataImport(Pattern, keys, rep):
    
    data = pd.DataFrame(columns=keys)
    
    for i in range(rep):
        mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, i)).item())
        
        temp = pd.Series()
        temp = temp.append(mat[keys].iloc[0])
        
        data = data.append(pd.DataFrame([temp.values],columns=keys), ignore_index=True)
                
    return data


def dataMerge(keys, rep):
    
    dataE = dataImport('Exponential', keys, rep)
    dataL = dataImport('Logistic', keys, rep)
    dataH = dataImport('Hill', keys, rep)

    data = pd.concat([dataE, dataL, dataH])
    data = data.set_index(np.arange(3*rep))
    
    return data


def dataProportion(data):
    
    n = len(data)
    
    data[data < 1] = 0
    
    for i in range(n):
        data.loc[i] = data.loc[i]/sum(data.loc[i])
        
    return data


def cloneNum(clone, data):
    
    dataE = data[clone][:300]
    dataL = data[clone][300:600]
    dataH = data[clone][600:]
    
    return dataE, dataL, dataH


def dataCombine(dataE, dataL, dataH):
    
    mat = pd.DataFrame(columns=['Pattern', 'Number'])
    
    mat['Pattern'] = ['Exponential']*len(dataE) + ['Logistic']*len(dataL) + ['Hill']*len(dataH)
    mat['Number'] = list(dataE) + list(dataL) + list(dataH)
    
    return mat


def boxPlot(data, n):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    sns.boxplot(x=data.columns[0], y=data.columns[1], data=data, width=0.5, palette=['lightcoral','darkgray','cadetblue'], linewidth=5)
    sns.swarmplot(x=data.columns[0], y=data.columns[1], data=data, color='lightsteelblue')
    
    ax.tick_params(axis='x',labelsize=30)
    ax.tick_params(axis='y',labelsize=30)
    
    ax.set_xlabel('Pattern', font1)
    ax.set_ylabel('Proportion', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)

    plt.savefig('box_%d.tif'%n, bbox_inches='tight')
    plt.savefig('box_%d.eps'%n, bbox_inches='tight')


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

X = dataMerge(keys, 300)
X = dataProportion(X)

# clone  'D:0.0,R:1.0,P:0.0'
clone = 'D:0.0,R:1.0,P:0.0'
dataE, dataL, dataH = cloneNum(clone, X)
data_1 = dataCombine(dataE, dataL, dataH)

boxPlot(data_1, 1)
p_1 = stats.kruskal(dataE, dataL, dataH)[1]

# clone  'D:0.1,R:0.8,P:0.1'
clone = 'D:0.1,R:0.8,P:0.1'
dataE, dataL, dataH = cloneNum(clone, X)
data_2 = dataCombine(dataE, dataL, dataH)

boxPlot(data_2, 2)
p_2 = stats.kruskal(dataE, dataL, dataH)[1]












