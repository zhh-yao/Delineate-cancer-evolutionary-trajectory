# In[1]: 
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]: 
## functions

def dataImport(Pattern, agent, keys):
    
    mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, agent), allow_pickle=True).item(), columns=keys)
                
    return mat


def dataSplit(mat):
    
    data_plus = 0.5*mat.values.T
    data_minus = -0.5*mat.values.T

    return data_plus, data_minus


def plot3(dataP1, dataM1, dataP2, dataM2, dataP3, dataM3, allcycle):
    
    colors = ['thistle','#FBA07E','yellowgreen','#72AC4C','lightskyblue','darkseagreen','#FCCC2E','tomato','violet','#7C99D2','peru','khaki','#F197B7','orange','#F5DEB3','tan']
    font1 = {'family':'Arial', 'weight':'normal', 'size':'60'}
    font2 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    time = np.arange(allcycle+1)
    
    fig, axs = plt.subplots(3, 1, figsize=(15,11), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)

    axs[0].stackplot(time, dataP1, colors=colors, rasterized=True)
    axs[0].stackplot(time, dataM1, colors=colors, rasterized=True)
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    #plt.text(-200, 70000000000,'Evolution 1', fontdict=font2)
    
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)

    axs[1].stackplot(time, dataP2, colors=colors, rasterized=True)
    axs[1].stackplot(time, dataM2, colors=colors, rasterized=True)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    #plt.text(-200, 40000000000,'Evolution 2', fontdict=font2)

    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['left'].set_visible(False)
    
    axs[2].stackplot(time, dataP3, colors=colors, rasterized=True)
    axs[2].stackplot(time, dataM3, colors=colors, rasterized=True)
    axs[2].set_xticks(np.arange(0, allcycle+1, 1000))
    axs[2].get_yaxis().set_visible(False)
    #plt.text(-200, 8000000000,'Evolution 3', fontdict=font2)
    
    axs[2].set_xlabel('Time (cell cycle)', font1, labelpad=20)
    axs[1].set_ylabel('Number of cells', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    axs[2].tick_params(length=10, labelsize=40, pad=20)
    
    plt.savefig('clone_plus.tif', bbox_inches='tight')
    plt.savefig('clone_plus.pdf', bbox_inches='tight')
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


para1 = [0.01, 0.008]
para2 = [0.01, 500000000000, 0.008]
para3 = [0.01, 1, 0.00000000001, 0.008]

dataH1 = dataImport('Hill', 8, keys)
dataHP1, dataHM1 = dataSplit(dataH1)

dataH2 = dataImport('Hill', 9, keys)
dataHP2, dataHM2 = dataSplit(dataH2)

dataH3 = dataImport('Hill', 10, keys)
dataHP3, dataHM3 = dataSplit(dataH3)


plot3(dataHP1, dataHM1, dataHP2, dataHM2, dataHP3, dataHM3, 3000)


dataL1 = dataImport('Logistic', 8, keys)
dataLP1, dataLM1 = dataSplit(dataL1)

dataL2 = dataImport('Logistic', 9, keys)
dataLP2, dataLM2 = dataSplit(dataL2)

dataL3 = dataImport('Logistic', 10, keys)
dataLP3, dataLM3 = dataSplit(dataL3)


plot3(dataLP1, dataLM1, dataLP2, dataLM2, dataLP3, dataLM3, 3000)


dataE1 = dataImport('Exponential', 8, keys)
dataEP1, dataEM1 = dataSplit(dataE1)

dataE2 = dataImport('Exponential', 9, keys)
dataEP2, dataEM2 = dataSplit(dataE2)

dataE3 = dataImport('Exponential', 10, keys)
dataEP3, dataEM3 = dataSplit(dataE3)


plot3(dataEP1, dataEM1, dataEP2, dataEM2, dataEP3, dataEM3, 3000)













