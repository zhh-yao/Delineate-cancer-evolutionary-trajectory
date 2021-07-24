# In[1]: 
# packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# In[2]: 
## functions
def WBC(Patient):
    
    rawData = pd.read_csv('CLL_Patients\WBC_counts\%d.txt' %Patient, skiprows=1, header=None, names = ['Year','WBC'])
    rawData['Year'] = rawData['Year']*365
    rawData['WBC'] = rawData['WBC']*1000000000
        
    return rawData


def dataImport(Patient, i, keys):
    
    temp = pd.read_csv('CLL_Patients\WBC_counts\%d.txt' %Patient, skiprows=1, header=None, names = ['Year','WBC(×10^9)'])
    timeI = math.ceil(temp['Year'][0]*365)
    
    computeData = pd.DataFrame(np.load('CLL_Patients\subclone_composition\evolution_%d_%d.npy' %(Patient, i), allow_pickle=True).item(), columns=keys)
    computeData.index = np.arange(timeI, timeI+len(computeData))
    
    return computeData


def plotDynamic(rowData, computeData, Patient):
    
    colors = ['thistle','#FBA07E','yellowgreen','#72AC4C','lightskyblue','darkseagreen','#FCCC2E','tomato','violet','#7C99D2','peru','khaki','#F197B7','orange','#F5DEB3','tan']
    
    time = np.arange(computeData.index[0], computeData.index[-1]+1, 1)
    
    fig, ax = plt.subplots(figsize=(15,4))
    ax.stackplot(time, computeData.T, colors=colors)
    
    plt.scatter(rowData['Year'], rowData['WBC'], color='k')
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)
    
    plt.savefig('CLL_%d.tif' %Patient, bbox_inches='tight')
    plt.savefig('CLL_%d.pdf' %Patient, bbox_inches='tight')
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

Patient = 1
rowData = WBC(Patient)
computeData = dataImport(Patient, 0, keys)

plotDynamic(rowData, computeData, Patient)

for Patient in range(1, 22):
    rowData = WBC(Patient)
    computeData = dataImport(Patient, 0, keys)
    plotDynamic(rowData, computeData, Patient)










