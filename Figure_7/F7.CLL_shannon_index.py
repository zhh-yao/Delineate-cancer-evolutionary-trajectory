# In[1]: 
# packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# In[2]: 

def dataImport(Patient, i, keys, time):
        
    all_mat = pd.DataFrame(np.load('CLL_Patients\subclone_composition\evolution_%d_%d.npy' %(Patient, i), allow_pickle=True).item(), columns=keys)
    
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


def indexGet(Patient, agent, keys, time):
    
    data = dataImport(Patient, agent, keys, time)
    composition = dataProportion(data)
    hi = shannon(composition)
    
    return hi


def indexList(Patient, keys, time, rep):
    
    hil = []
    
    for agent in range(rep):
        hil.append(indexGet(Patient, agent, keys, time))
    
    return hil


def dataCombine(Patients, keys, time, rep):
    
    PList = []
    HList = []
    
    mat = pd.DataFrame(columns=['Status', 'Heterogeneity'])
    
    for i in Patients[0]:
        
        PList = PList + ['Unmutated']*rep
        HList = HList + indexList(i, keys, time, rep)
    
    for j in Patients[1]:
        
        PList = PList + ['Mutated']*rep
        HList = HList + indexList(j, keys, time, rep)
    
    mat['Status'] = PList
    mat['Heterogeneity'] = HList
    
    return mat


def boxPlot(data):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    sns.boxplot(x=data.columns[0], y=data.columns[1], data=data, width=0.5, palette=['lightskyblue','thistle'])
    sns.swarmplot(x=data.columns[0], y=data.columns[1], data=data, color='lightsteelblue')
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)

    plt.savefig('heterogeneity.tif', bbox_inches='tight')
    plt.savefig('heterogeneity.eps', bbox_inches='tight')



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

# unmutated IGHV  vs  mutated IGHV
Patients = [[1,3,5,6,11,12,13,14,15,21], [2,4,7,8,9,10,16,17,18,19,20]]

barData = dataCombine(Patients, keys, 0, 40)

boxPlot(barData)

p1 = stats.mannwhitneyu(barData['Heterogeneity'][:400], barData['Heterogeneity'][400:])[1]

# unmutated IGHV  vs  mutated IGHV
Patients = [[1,6], [4,10,18,19]]

barData = dataCombine(Patients, keys, 0, 40)

boxPlot(barData)

p2 = stats.mannwhitneyu(barData['Heterogeneity'][:80], barData['Heterogeneity'][80:])[1]


# In[4]: 

def dataCombine3(Patients, keys, time, rep):
    
    PList = []
    HList = []
    
    mat = pd.DataFrame(columns=['Pattern', 'Heterogeneity'])
    
    for i in Patients[0]:
        
        PList = PList + ['Logistic']*rep
        HList = HList + indexList(i, keys, time, rep)
    
    for j in Patients[1]:
        
        PList = PList + ['Indeterminate']*rep
        HList = HList + indexList(j, keys, time, rep)
    
    for k in Patients[2]:
        
        PList = PList + ['Exponential']*rep
        HList = HList + indexList(k, keys, time, rep)
    
    mat['Pattern'] = PList
    mat['Heterogeneity'] = HList
    
    return mat


def boxPlot(data):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    sns.boxplot(x=data.columns[0], y=data.columns[1], data=data, width=0.5, palette=['darkgray','khaki','lightcoral'])
    sns.swarmplot(x=data.columns[0], y=data.columns[1], data=data, color='lightsteelblue')
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)

    plt.savefig('heterogeneity.tif', bbox_inches='tight')
    plt.savefig('heterogeneity.eps', bbox_inches='tight')



# LOG, IND, EXP
Patients = [[16,4,19,18,11], [7,17,10,8,12,15], [20,2,9,3,1,6,5,14,13,21]]

barData = dataCombine3(Patients, keys, 0, 40)

boxPlot(barData)











