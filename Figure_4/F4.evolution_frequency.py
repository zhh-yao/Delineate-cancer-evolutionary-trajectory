# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]: 
def dataImport(Pattern, agent, keys):
    
    mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, agent), allow_pickle=True).item(), columns=keys)
                
    return mat


def mat2bool(mat):
    
    for i in range(len(mat)):
        temp = []
        matR = mat.loc[i]
        for j in matR:
            if j >= 1:
                temp.append(1)
            else:
                temp.append(0)
        
        mat.loc[i] = temp
        
    return mat


def diff_num(mat):
    
    inc = []
    dec = []
    n = len(mat)
    
    for cycle in range(1, n):
        
        diff = mat.loc[cycle] - mat.loc[cycle-1]
        
        inc.append(len(np.where(diff == 1)[0]))
        dec.append(len(np.where(diff == -1)[0]))
        
    return inc, dec


def all_diff_num(Pattern, rep, keys):
    
    all_inc = 0
    all_dec = 0
    
    for agent in range(rep):
        mat = dataImport(Pattern, agent, keys)
        mat = mat2bool(mat)
        inc, dec = diff_num(mat)
        
        all_inc += np.array(inc)
        all_dec += np.array(dec)
        
    return all_inc, all_dec


def sum100(data):
    
    num = []
    for gap in range(0, len(data), 100):
        
        temp = sum(data[gap:gap+100])
        num.append(temp)
        
    return num


def sub_plot(num_inc, num_dec, Pattern):
    
    colors = {'Exponential':'lightcoral', 'Logistic':'darkgray', 'Hill':'cadetblue'}
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    n = len(num_inc)
    x = np.arange(100, (n+1)*100, 100)
    
    plt.bar(x, num_inc, width=60, color=colors[Pattern])
    plt.bar(x, -np.array(num_dec), width=60, color='w', edgecolor=colors[Pattern], linewidth=2, hatch='/')
    
    plt.axhline(y=0, lw=1, color='k')
    
    mean_inc = np.mean(num_inc[0:20])
    plt.axhline(y=mean_inc, lw=3, color=colors[Pattern], linestyle='--')
    plt.text(1700, mean_inc+0.6,'mean=%.2f'%mean_inc, fontdict=font1)
    mean_dec = -np.mean(num_dec[0:20])
    plt.axhline(y=mean_dec, lw=3, color=colors[Pattern], alpha=0.5, linestyle='--')
    plt.text(100, mean_dec-0.8,'mean=%.2f'%mean_dec, fontdict=font1)

    plt.xlim([0, 3090])
    plt.ylim([-2.9, 2.9])
    
    ax.set_xlabel('Time (cell cycle)', font1, labelpad=20)
    ax.set_ylabel('Frequency', font1, labelpad=20)
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)
        
    plt.savefig('freq_%s.tif' %Pattern, bbox_inches='tight')
    plt.savefig('freq_%s.pdf' %Pattern, bbox_inches='tight')
    plt.show()


# In[3]: plot

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

all_inc_E, all_dec_E = all_diff_num('Exponential', 200, keys)
all_inc_L, all_dec_L = all_diff_num('Logistic', 200, keys)
all_inc_H, all_dec_H = all_diff_num('Hill', 200, keys)

num_inc_E, num_dec_E = np.array(sum100(all_inc_E))/200,  np.array(sum100(all_dec_E))/200
num_inc_L, num_dec_L = np.array(sum100(all_inc_L))/200,  np.array(sum100(all_dec_L))/200
num_inc_H, num_dec_H = np.array(sum100(all_inc_H))/200,  np.array(sum100(all_dec_H))/200

sub_plot(num_inc_E, num_dec_E, 'Exponential')
sub_plot(num_inc_L, num_dec_L, 'Logistic')
sub_plot(num_inc_H, num_dec_H, 'Hill')














