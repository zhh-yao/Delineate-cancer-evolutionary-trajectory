# In[1]: 
# packages
import numpy as np
import pandas as pd
from scipy.integrate import odeint
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


def plot(data, allcycle, Pattern, agent):
    
    #colors = ['thistle','rosybrown','yellowgreen','#9ECCD2','lightskyblue','darkseagreen','cornflowerblue','tomato','violet','plum','peru','khaki','steelblue','orange','#F1999D','tan']
    colors = ['thistle','#FBA07E','yellowgreen','#72AC4C','lightskyblue','darkseagreen','#FCCC2E','tomato','violet','#7C99D2','peru','khaki','#F197B7','orange','#F5DEB3','tan']
    font1 = {'family':'Arial', 'weight':'normal', 'size':'60'}
    font2 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    time = np.arange(allcycle+1)
    
    fig, ax = plt.subplots(figsize=(13,11))
    ax.stackplot(time, data, colors=colors, rasterized=True)
    
    if Pattern == 'Exponential':
        t1 = np.arange(allcycle+1)
        y = odeint(Exponential, 5000000000, t1, args = (para1,))
        ax.plot(t1, y, linewidth=10, ls='--', color='lightcoral', label='Aggressive')
    elif Pattern == 'Logistic':
        t2 = np.arange(allcycle+1)
        y = odeint(Logistic, 5000000000, t2, args = (para2,))
        ax.plot(t2, y, linewidth=10, ls='--', color='darkgray', label='Bounded')
    elif Pattern == 'Hill':
        t3 = np.arange(allcycle+1)
        y = odeint(Hill, 5000000000, t3, args = (para3,))
        ax.plot(t3, y, linewidth=10, ls='--', color='cadetblue', label='Indolent')

    ax.set_xlabel('Time (cell cycle)', font1, labelpad=20)
    
    if Pattern == 'Hill':
        ax.set_ylabel('Number of cells (×10¹⁰)', font1, labelpad=20)
    elif Pattern == 'Logistic':
        ax.set_ylabel('Number of cells (×10¹¹)', font1, labelpad=20)
    elif Pattern == 'Exponential':
        ax.set_ylabel('Number of cells (×10¹²)', font1, labelpad=20)
    
    plt.xlim([0, 3000])
    #plt.ylim([0, 200000000000])
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    ax.legend(loc='upper left', prop=font2, frameon=False)
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)
    
    plt.savefig('clone_%s_%d.tif' %(Pattern, agent), bbox_inches='tight')
    plt.savefig('clone_%s_%d.pdf' %(Pattern, agent), bbox_inches='tight')
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


Pattern = 'Exponential'
allcycle = 3000
agent = 0
data = dataImport(Pattern, agent, keys)
plot(data.values.T, allcycle, Pattern, agent)


Pattern = 'Logistic'
allcycle = 3000
agent = 0
data = dataImport(Pattern, agent, keys)
plot(data.values.T, allcycle, Pattern, agent)


Pattern = 'Hill'
allcycle = 3000
agent = 0
data = dataImport(Pattern, agent, keys)
plot(data.values.T, allcycle, Pattern, agent)















