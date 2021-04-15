# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


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


def prolifPlot(data, Pattern):
    
    colors = {'Exponential':'lightcoral', 'Logistic':'darkgray', 'Hill':'cadetblue'}
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'40'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    plt.hist(data, bins=50, color=colors[Pattern])
    
    mean = np.mean(data)
    ax.axvline(mean, lw=4, ls='--', color='k')
    plt.text(mean+0.02, 15, 'mean=%.2f'%mean, fontdict=font2)
    
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    
    ax.set_xlabel('Rate of Proliferation', font1)
    ax.set_ylabel('Frequency', font1)
    
    plt.xlim([-0.05, 0.55])
    plt.ylim([0, 18])
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)
    
    plt.savefig('prolif_%s.tif'%Pattern, bbox_inches='tight')
    plt.savefig('prolif_%s.eps'%Pattern, bbox_inches='tight')


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


pList_E_0 = prolifList('Exponential', 300, allclones, 0)
pList_L_0 = prolifList('Logistic', 300, allclones, 0)
pList_H_0 = prolifList('Hill', 300, allclones, 0)

prolifPlot(pList_E_0, 'Exponential')
prolifPlot(pList_L_0, 'Logistic')
prolifPlot(pList_H_0, 'Hill')

p_0 = stats.kruskal(pList_E_0, pList_L_0, pList_H_0)[1]


pList_E_2000 = prolifList('Exponential', 300, allclones, 2000)
pList_L_2000 = prolifList('Logistic', 300, allclones, 2000)
pList_H_2000 = prolifList('Hill', 300, allclones, 2000)

prolifPlot(pList_E_2000, 'Exponential')
prolifPlot(pList_L_2000, 'Logistic')
prolifPlot(pList_H_2000, 'Hill')

p_2000 = stats.kruskal(pList_E_2000, pList_L_2000, pList_H_2000)[1]









