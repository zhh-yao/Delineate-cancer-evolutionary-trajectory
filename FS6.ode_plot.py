import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint


def exponential(x, t, para):

    r,a = para
    
    return np.array(r*x-a*x)


def logistic(x, t, para):

    r,k,a = para
    
    return np.array(r*x*(1-x/k)-a*x)


def hill(x, t, para):
    
    r,b,c,a = para
    
    return np.array(r*x**b/(1+c*x**b)-a*x)


def plot(para1, para2, para3):
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    tE = np.arange(0, 2000, 1)
    tL = np.arange(0, 3500, 1)
    tH = np.arange(0, 4000, 1)
    
    y = odeint(exponential, 5000000000, tE, args = (para1,))
    ax.plot(tE, y, linewidth=1, color='lightcoral', label='Exponential')    
    y = odeint(logistic, 5000000000, tL, args = (para2,))
    ax.plot(tL, y, linewidth=1, color='darkgray', label='Logistic')    
    y = odeint(hill, 5000000000, tH, args = (para3,))
    ax.plot(tH, y, linewidth=1, color='cadetblue', label='Hill')    
    
    for i in range(300):
        random.seed(i)
        paraE = [np.random.normal(para1[0], 0.5*para1[0]**2), para1[1]]
        y = odeint(exponential, 5000000000, tE, args = (paraE,))
        ax.plot(tE, y, linewidth=1, color='lightcoral')
    
        paraL = [np.random.normal(para2[0], 0.5*para2[0]**2), para2[1], para2[2]]
        y = odeint(logistic, 5000000000, tL, args = (paraL,))
        ax.plot(tL, y, linewidth=1, color='darkgray')

        paraH = [np.random.normal(para3[0], 0.5*para3[0]**2), para3[1], para3[2], para3[3]]
        y = odeint(hill, 5000000000, tH, args = (paraH,))
        ax.plot(tH, y, linewidth=1, color='cadetblue')
        
    font1 = {'family':'Arial', 'weight':'normal', 'size':'45'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'35'}
    ax.set_xlabel('Time (Cell Cycle)', font1)
    ax.set_ylabel('Number of cells (×10¹¹)', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(length=10, labelsize=30)
    
    l = ax.legend(markerscale=10, loc='best', prop=font2, frameon=False)
    for i in l.legendHandles:
        i.set_linewidth(10)
        
    plt.savefig('ode.tif', bbox_inches='tight')
    plt.savefig('ode.eps', bbox_inches='tight')
    plt.show()


para1 = [0.01, 0.008]                     #exponential
para2 = [0.01, 500000000000, 0.008]       #logistic
para3 = [0.01, 1, 0.00000000001, 0.008]   #hill
plot(para1, para2, para3)


