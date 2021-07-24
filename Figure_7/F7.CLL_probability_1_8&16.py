# In[1]:

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]: Agent analysis: proliferation

def dataImport(Patient, i, keys):
    
    mat = pd.DataFrame(np.load('CLL_Patients\subclone_composition\evolution_%d_%d.npy' %(Patient, i), allow_pickle=True).item(), columns=keys)
                
    return mat


def prolifCalculate(data):
    
    proList = []    

    for i in range(len(data)):
        pi = 0
        
        for c in data.columns:
            pi += float(c[14:17]) * data[c][i]
        
        pi = pi/sum(data.iloc[i,:])

        proList.append(pi)

    return proList


def prolifCombine(Patient, keys, rep):
    
     prolifMat = pd.DataFrame()
     
     for i in range(rep):
         data = dataImport(Patient, i, keys)
         prolifMat['%d'%i] = prolifCalculate(data)
    
     return prolifMat


def mean_confidence_interval(data, confidence=0.95):
    
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def rateMCI(rateMat):
    
    mean, ciu, cil = [], [], []
    
    for r in range(len(rateMat)):
        
        m, c1, c2 = mean_confidence_interval(rateMat.iloc[r,:], confidence=0.95)

        mean.append(m)
        ciu.append(c1)
        cil.append(c2)
        
    return mean, ciu, cil


def proPlot(p1, p16):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'normal', 'size':'40'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    x = np.arange(0, 2000, 1)
    ax.fill_between(x, p1[1][:2000], p1[2][:2000], color='lightskyblue', alpha=0.5)
    ax.plot(x, p1[0][:2000], lw=1, color='lightskyblue', label='Patient %d'%1)      
    
    ax.fill_between(x, p16[1][:2000], p16[2][:2000], color='thistle', alpha=0.5)
    ax.plot(x, p16[0][:2000], lw=1, color='thistle', label='Patient %d'%16)      
    
    ax.tick_params(axis='x',labelsize=20)
    ax.tick_params(axis='y',labelsize=30)
    
    ax.set_xlabel('Time (Days)', font1, labelpad=20)
    ax.set_ylabel('Probability of proliferation', font1, labelpad=20)
    
    plt.ylim([0, 0.4])

    ax.legend(loc=4, prop=font2, frameon=False)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
        
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, labelsize=40, pad=20)

    plt.savefig('Proliferation.tif', bbox_inches='tight')
    plt.savefig('Proliferation.pdf', bbox_inches='tight')


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

# Patient 1, 16: all of them have SF3B1 mutation, Patient 1 got another 11q23 deletion
##Patient 1
Patient = 1
prolifMat1 = prolifCombine(Patient, keys, 40)
p1 = rateMCI(prolifMat1)

##Patient 16
Patient = 16
prolifMat16 = prolifCombine(Patient, keys, 40)
p16 = rateMCI(prolifMat16)

proPlot(p1, p16)


# In[3]: Agent analysis: resting

def dataImport(Patient, i, keys):
    
    mat = pd.DataFrame(np.load('evolution_%d_%d.npy' %(Patient, i), allow_pickle=True).item(), columns=keys)
                
    return mat


def restCalculate(data):
    
    restList = []    

    for i in range(len(data)):
        pi = 0
        
        for c in data.columns:
            pi += float(c[8:11]) * data[c][i]
        
        pi = pi/sum(data.iloc[i,:])

        restList.append(pi)

    return restList


def restCombine(Patient, keys, rep):
    
     restMat = pd.DataFrame()
     
     for i in range(rep):
         data = dataImport(Patient, i, keys)
         restMat['%d'%i] = restCalculate(data)
    
     return restMat


def mean_confidence_interval(data, confidence=0.95):
    
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def rateMCI(rateMat):
    
    mean, ciu, cil = [], [], []
    
    for r in range(len(rateMat)):
        
        m, c1, c2 = mean_confidence_interval(rateMat.iloc[r,:], confidence=0.95)

        mean.append(m)
        ciu.append(c1)
        cil.append(c2)
        
    return mean, ciu, cil


def restPlot(r1, r8, r16):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'30'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    
    x1 = np.arange(0, len(r1[0]), 1)
    ax.fill_between(x1, r1[1], r1[2], alpha=0.3)
    ax.plot(x1, r1[0], lw=5, label='Patient %d'%1)      
    
    x8 = np.arange(0, len(r8[0]), 1)
    ax.fill_between(x8, r8[1], r8[2], alpha=0.3)
    ax.plot(x8, r8[0], lw=5, label='Patient %d'%8)      
    
    x16 = np.arange(0, len(r16[0]), 1)
    ax.fill_between(x16, r16[1], r16[2], alpha=0.3)
    ax.plot(x16, r16[0], lw=5, label='Patient %d'%16)      
    
    ax.tick_params(axis='x',labelsize=20)
    ax.tick_params(axis='y',labelsize=30)
    
    ax.set_xlabel('Time (Days)', font1)
    ax.set_ylabel('Rate of resting', font1)
    
    ax.legend(loc='best', prop=font2, frameon=False)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.savefig('Rest.tif', bbox_inches='tight')
    plt.savefig('Rest.eps', bbox_inches='tight')


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

# Patient 1, 8, 16: all of them have SF3B1 mutation, Patient 1 got another 11q23 deletion
##Patient 1
Patient = 1
restMat1 = restCombine(Patient, keys, 10)
r1 = rateMCI(restMat1)
##Patient 8
Patient = 8
restMat8 = restCombine(Patient, keys, 10)
r8 = rateMCI(restMat8)
##Patient 16
Patient = 16
restMat16 = restCombine(Patient, keys, 10)
r16 = rateMCI(restMat16)

restPlot(r1, r8, r16)


