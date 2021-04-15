# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]: 

def dataImport(Pattern, agent, keys):
    
    mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, agent), allow_pickle=True).item(), columns=keys)
                
    return mat


def prolifCalculate(mat, time):
    
    prolif = 0
    data = mat.loc[time]
    
    for i in range(len(data)):
        prolif += float(data.index[i][14:17]) * data[i]
        
    prolif = prolif/sum(data)

    return prolif


def prolifList(Pattern, rep, keys, time):
    
    pList = []
    
    for agent in range(rep):
        
        mat = dataImport(Pattern, agent, keys)
        prolif = prolifCalculate(mat, time)
        
        pList.append(prolif)
        
    return pList


def meanProlif(Pattern, rep, keys, cycle):
    
    meanP = []
    
    for time in np.arange(0, cycle+1, 100):
        
        temp = np.mean(prolifList(Pattern, rep, keys, time))
        meanP.append(temp)
        
    return meanP


def prolifPlot(dataE, dataL, dataH):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'40'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    x = np.arange(0, len(dataE)*100, 100)
    
    ax.plot(x, dataE, lw=10, label='Exponential', color='lightcoral')
    ax.plot(x, dataL, lw=10, label='Logistic', color='darkgray')
    ax.plot(x, dataH, lw=10, label='Hill', color='cadetblue')
    
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    
    ax.set_xlabel('Time (Cell Cycle)', font1)
    ax.set_ylabel('Rate of Proliferation', font1)
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)
    
    ax.legend(loc='best', prop=font2, frameon=False)
    
    plt.savefig('prolif_0_2000.tif', bbox_inches='tight')
    plt.savefig('prolif_0_2000.eps', bbox_inches='tight')


# In[3]: 

allclones =['D:0.0,R:0.9,P:0.1',
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


meanP_E = meanProlif('Exponential', 300, allclones, 2000)
meanP_L = meanProlif('Logistic', 300, allclones, 2000)
meanP_H = meanProlif('Hill', 300, allclones, 2000)

prolifPlot(meanP_E, meanP_L, meanP_H)







