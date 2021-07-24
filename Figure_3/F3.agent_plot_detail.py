# In[1]: 
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[2]: 
## functions
def Exponential(x, t, Para):

    r,a = Para
    
    return np.array(r*x-a*x)


def Logistic(x, t, para):

    r,k,a = para
    
    return np.array(r*x*(1-x/k)-a*x)


def Hill(x, t, para):
    
    r,b,c,a = para
    
    return np.array(r*x**b/(1+c*x**b)-a*x)


def dataImport(Pattern, agent, keys):
    
    mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, agent), allow_pickle=True).item(), columns=keys)
                
    return mat


def plot3(dataE, dataL, dataH, allcycle, agent):
    
    colors = ['thistle','lightskyblue']
    font1 = {'family':'Arial', 'weight':'normal', 'size':'60'}
    font2 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    time = np.arange(allcycle+1)
    
    fig, axs = plt.subplots(3, 1, figsize=(13,11), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    axs[0].stackplot(time, dataE, colors=colors, rasterized=True)
    axs[0].get_xaxis().set_visible(False)
    axs[0].set_yticks(np.arange(0, 5000000000, 2000000000))
    axs[0].set_ylim(0, 5000000000)
    plt.text(50, 14500000000,'Aggressive', fontdict=font2)

    axs[1].stackplot(time, dataL, colors=colors, rasterized=True)
    axs[1].get_xaxis().set_visible(False)
    axs[1].set_yticks(np.arange(0, 5000000000, 2000000000))
    axs[1].set_ylim(0, 5000000000)
    plt.text(50, 9000000000,'Bounded', fontdict=font2)
    
    axs[2].stackplot(time, dataH, colors=colors, rasterized=True)
    axs[2].set_xticks(np.arange(0, allcycle+1, 1000))
    axs[2].set_yticks(np.arange(0, 5000000000, 2000000000))
    axs[2].set_ylim(0, 5000000000)
    plt.text(50, 3500000000,'Indolent', fontdict=font2)
    
    axs[2].set_xlabel('Time (cell cycle)', font1, labelpad=20)
    axs[1].set_ylabel('Number of cells (×10⁹)', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    labels = axs[0].get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    axs[0].tick_params(length=10, labelsize=40, pad=20)
    
    labels = axs[1].get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    axs[1].tick_params(length=10, labelsize=40, pad=20)
    
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    axs[2].tick_params(length=10, labelsize=40, pad=20)
    
    
    plt.savefig('clone_detail_%d.tif' %agent, bbox_inches='tight')
    plt.savefig('clone_detail_%d.pdf' %agent, bbox_inches='tight')
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


allcycle = 3000
agent = 0

Pattern = 'Exponential'
data = dataImport(Pattern, agent, keys)
dataE = data[['D:0.0,R:0.9,P:0.1', 'D:0.1,R:0.9,P:0.0']].values.T


Pattern = 'Logistic'
data = dataImport(Pattern, agent, keys)
dataL = data[['D:0.0,R:0.9,P:0.1', 'D:0.1,R:0.9,P:0.0']].values.T


Pattern = 'Hill'
data = dataImport(Pattern, agent, keys)
dataH = data[['D:0.0,R:0.9,P:0.1', 'D:0.1,R:0.9,P:0.0']].values.T


plot3(dataE, dataL, dataH, 3000, 0)













