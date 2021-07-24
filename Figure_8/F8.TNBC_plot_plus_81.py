# In[1]: 
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]: 
## functions

def dataImport(Patient, Pattern, agent, keys):
    
    mat = pd.DataFrame(np.load('Pt%s_%d_%d.npy'%(Patient, Pattern, agent), allow_pickle=True).item(), columns=keys)
                
    return mat


def dataSplit(mat):
    
    data_plus = 0.5*mat.values.T
    data_minus = -0.5*mat.values.T

    return data_plus, data_minus


def plot(Patient, data, allcycle):
    
    colors = ['thistle','#FBA07E','yellowgreen','#72AC4C','lightskyblue','darkseagreen','#FCCC2E','tomato','violet','#7C99D2','peru','khaki','#F197B7','orange','#F5DEB3','tan']
    font1 = {'family':'Arial', 'weight':'normal', 'size':'60'}
    time = np.arange(allcycle+1)
    
    n = len(data)
    
    fig, axs = plt.subplots(n, 1, figsize=(15,4*n), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    for i in range(n-1):
        
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
    
        axs[i].stackplot(time, data[i][0], colors=colors, rasterized=True)
        axs[i].stackplot(time, data[i][1], colors=colors, rasterized=True)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].set_ylim(-6000000000, 6000000000)
        
    axs[n-1].spines['top'].set_visible(False)
    axs[n-1].spines['right'].set_visible(False)
    axs[n-1].spines['left'].set_visible(False)
    
    axs[n-1].stackplot(time, data[n-1][0], colors=colors, rasterized=True)
    axs[n-1].stackplot(time, data[n-1][1], colors=colors, rasterized=True)
    axs[n-1].set_xticks(np.arange(0, allcycle+1, 1000))
    axs[n-1].get_yaxis().set_visible(False)
    axs[n-1].set_ylim(-6000000000, 6000000000)
    
    axs[n-1].set_xlabel('Time (cell cycle)', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    labels = axs[n-1].get_xticklabels() + axs[n-1].get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    axs[n-1].tick_params(length=10, labelsize=40, pad=20)
    
    plt.savefig('TNBC_%s_plus.tif' %Patient, bbox_inches='tight')
    plt.savefig('TNBC_%s_plus.pdf' %Patient, bbox_inches='tight')
    plt.show()


# In[3]: 
## plot subclonal dynamics through total number of cells
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



dataE = []
for i in range(5):
    temp = dataImport('081', 1, i, keys)
    dataE.append(dataSplit(temp))

plot('081', dataE, 2000)



dataL = []
for i in range(5):
    temp = dataImport('081', 2, i, keys)
    dataL.append(dataSplit(temp))

plot('081', dataL, 2000)



dataH = []
for i in range(5):
    temp = dataImport('081', 3, i, keys)
    dataH.append(dataSplit(temp))

plot('081', dataH, 2000)














