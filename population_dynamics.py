import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def Exponential(x, t, Para):

    r,a = Para
    
    return np.array(r*x-a*x)


def Logistic(x, t, para):

    r,k,a = para
    
    return np.array(r*x*(1-x/k)-a*x)


def Hill(x, t, para):
    
    r,b,c,a = para
    
    return np.array(r*x**b/(1+c*x**b)-a*x)


def CellNumber(Cycle, model, Para, N0=5000000000):
    
    Times = np.arange(0, Cycle+1)
    CellNum = odeint(model, N0, Times, args = (Para,))
    
    return CellNum


def plot(para1, para2, para3):
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'60'}
    font2 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    tE = np.arange(0, 2000, 1)
    tL = np.arange(0, 3000, 1)
    tH = np.arange(0, 3000, 1)
    
    y = odeint(Exponential, 5000000000, tE, args = (para1,))
    ax.plot(tE, y, linewidth=15, color='lightcoral', label='Aggressive')    
    y = odeint(Logistic, 5000000000, tL, args = (para2,))
    ax.plot(tL, y, linewidth=15, color='darkgray', label='Bounded')    
    y = odeint(Hill, 5000000000, tH, args = (para3,))
    ax.plot(tH, y, linewidth=15, color='cadetblue', label='Indolent')    
    
    plt.xlim([0, 3000])
    plt.ylim([0, 300000000000])
    
    ax.set_xlabel('Time (Cell Cycle)', font1, labelpad=20)
    ax.set_ylabel('Number of cells (×10¹¹)', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)
    
    l = ax.legend(markerscale=10, loc='best', prop=font2, frameon=False)
    for i in l.legendHandles:
        i.set_linewidth(10)
        
    #plt.savefig('ode.tif', bbox_inches='tight')
    #plt.savefig('ode.eps', bbox_inches='tight')
    plt.show()


para1 = [0.01, 0.008]                     #exponential
para2 = [0.01, 500000000000, 0.008]       #logistic
para3 = [0.01, 1, 0.00000000001, 0.008]   #hill
plot(para1, para2, para3)

