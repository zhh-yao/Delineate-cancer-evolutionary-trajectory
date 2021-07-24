# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[2]:

def dataImport(Patient, Prob):
    
    data = pd.read_csv('TNBC_Patients\Probability\PT%s_%s.csv' %(Patient, Prob), names=['Cell', 'Probability'])
    
    return data


def normalize(data1, data2):
    
    temp = max(data1['Probability']) + max(data2['Probability'])
        
    nor_data1, nor_data2 = (data1['Probability']-min(data1['Probability']))/temp, (data2['Probability']-min(data2['Probability']))/temp
    
    return nor_data1, nor_data2


def split(data_apop, data_prol, p_a, p_p):
    
    ind11 = np.where(data_apop >= (p_a-0.05))[0]
    ind12 = np.where(data_apop < (p_a+0.05))[0]
    ind1 = [new for new in ind11 if new in ind12]
    
    ind21 = np.where(data_prol >= (p_p-0.05))[0]
    ind22 = np.where(data_prol < (p_p+0.05))[0]
    ind2 = [new for new in ind21 if new in ind22]
    
    ind = [new for new in ind1 if new in ind2]

    return ind

'''
def plot(Patient, data1, data2, ind1, ind2):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    ax.scatter(data1, data2, color='steelblue')
    ax.scatter(data1[ind1], data2[ind1], color='orange', label='A')
    ax.scatter(data1[ind2], data2[ind2], color='plum', label='B')
    ax.set_aspect(1.)
    
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    
    ax_histx.spines['left'].set_linewidth(2)
    ax_histx.spines['right'].set_linewidth(2)
    ax_histx.spines['top'].set_linewidth(2)
    ax_histx.spines['bottom'].set_linewidth(2)

    ax_histy.spines['left'].set_linewidth(2)
    ax_histy.spines['right'].set_linewidth(2)
    ax_histy.spines['top'].set_linewidth(2)
    ax_histy.spines['bottom'].set_linewidth(2)

    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    ax_histx.hist(data1, bins=100, color='steelblue')
    ax_histy.hist(data2, bins=50, orientation='horizontal', color='steelblue')
    ax_histx.hist(data1[ind1], bins=100, color='orange')
    ax_histy.hist(data2[ind1], bins=50, orientation='horizontal', color='orange')
    ax_histx.hist(data1[ind2], bins=100, color='plum')
    ax_histy.hist(data2[ind2], bins=50, orientation='horizontal', color='plum')
    
    #ax.set_xlim(-0.02, 0.32)
    #ax.set_ylim(0.05, 0.22)
    ax.set_xlabel('Proliferation', font1)
    ax.set_ylabel('Apoptosis', font1)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, width=2, labelsize=40)
    labels = ax_histx.get_xticklabels() + ax_histx.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax_histx.tick_params(length=10, width=2, labelsize=40)
    labels = ax_histy.get_xticklabels() + ax_histy.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax_histy.tick_params(length=10, width=2, labelsize=40)
      
    plt.savefig('Pt%s_probability.tif' %Patient, bbox_inches='tight')
    plt.savefig('Pt%s_probability.pdf' %Patient, bbox_inches='tight')
    plt.show()
'''

def plot(Patient, data1, data2, ind1, ind2, ind3):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    
    fig, ax = plt.subplots(figsize=(13,11))
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    ax.scatter(data1, data2, color='#A6A6A6', label='Others')
    ax.scatter(data1[ind1], data2[ind1], marker='*', s=200, color='#FBA07E')  # 001000
    ax.scatter(data1[ind3], data2[ind3], marker='*', s=200, color='#72AC4C')  # 010801
    #ax.scatter(data1[ind2], data2[ind2], marker='*', s=200, color='#FCCC2E')  # 020602
    #ax.scatter(data1[ind2], data2[ind2], marker='*', s=200, color='#7C99D2')  # 030403
    #ax.scatter(data1[ind2], data2[ind2], marker='*', s=200, color='#F197B7')  # 040204
    ax.scatter(data1[ind2], data2[ind2], marker='*', s=200, color='#F5DEB3')  # 050005
    ax.set_aspect(1.)
    
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-0.02, 0.45)
    ax.set_xlabel('Proliferation', font1, labelpad=20)
    ax.set_ylabel('Apoptosis', font1, labelpad=20)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(length=10, width=1, labelsize=40, pad=20)
    
    #plt.legend(loc='best', prop=font1, frameon=False)
    
    plt.savefig('Pt%s_probability.tif' %Patient, bbox_inches='tight')
    plt.savefig('Pt%s_probability.pdf' %Patient, bbox_inches='tight')
    plt.show()


# In[3]:
'''
test 1
010801.0, 020602.0
'''
Patient = '039'

data_Apop_039 = dataImport(Patient, 'Apoptosis')
data_Prol_039 = dataImport(Patient, 'Proliferation')

Pt39_Prol, Pt39_Apop = normalize(data_Prol_039, data_Apop_039)
ind1 = split(Pt39_Apop, Pt39_Prol, 0, 0)
ind2 = split(Pt39_Apop, Pt39_Prol, 0.1, 0.1)
ind3 = split(Pt39_Apop, Pt39_Prol, 0.2, 0.2)

plot(Patient, Pt39_Prol, Pt39_Apop, ind1, ind2, ind3)

p39_1 = len(ind1)/len(Pt39_Prol)
p39_2 = len(ind2)/len(Pt39_Prol)
p39_3 = len(ind3)/len(Pt39_Prol)

'''
test 2
001000.0, 030403.0
'''
Patient = '058'

data_Apop_058 = dataImport(Patient, 'Apoptosis')
data_Prol_058 = dataImport(Patient, 'Proliferation')

Pt58_Prol, Pt58_Apop = normalize(data_Prol_058, data_Apop_058)
ind1 = split(Pt58_Apop, Pt58_Prol, 0, 0)
ind2 = split(Pt58_Apop, Pt58_Prol, 0.3, 0.3)
ind3 = split(Pt58_Apop, Pt58_Prol, 0.4, 0.4)

plot(Patient, Pt58_Prol, Pt58_Apop, ind1, ind2, ind3)

p58_1 = len(ind1)/len(Pt58_Prol)
p58_2 = len(ind2)/len(Pt58_Prol)
p58_3 = len(ind3)/len(Pt58_Prol)


Patient = '089'

data_Apop_089 = dataImport(Patient, 'Apoptosis')
data_Prol_089 = dataImport(Patient, 'Proliferation')

Pt89_Prol, Pt89_Apop = normalize(data_Prol_089, data_Apop_089)
ind1 = split(Pt89_Apop, Pt89_Prol, 0, 0)
ind2 = split(Pt89_Apop, Pt89_Prol, 0.3, 0.3)
ind3 = split(Pt89_Apop, Pt89_Prol, 0.4, 0.4)

plot(Patient, Pt89_Prol, Pt89_Apop, ind1, ind2, ind3)

p89_1 = len(ind1)/len(Pt89_Prol)
p89_2 = len(ind2)/len(Pt89_Prol)
p89_3 = len(ind3)/len(Pt89_Prol)

Patient = '126'

data_Apop_126 = dataImport(Patient, 'Apoptosis')
data_Prol_126 = dataImport(Patient, 'Proliferation')

Pt126_Prol, Pt126_Apop = normalize(data_Prol_126, data_Apop_126)
ind1 = split(Pt126_Apop, Pt126_Prol, 0, 0)
ind2 = split(Pt126_Apop, Pt126_Prol, 0.3, 0.3)
ind3 = split(Pt126_Apop, Pt126_Prol, 0.4, 0.4)

plot(Patient, Pt126_Prol, Pt126_Apop, ind1, ind2, ind3)

p126_1 = len(ind1)/len(Pt126_Prol)
p126_2 = len(ind2)/len(Pt126_Prol)
p126_3 = len(ind3)/len(Pt126_Prol)

'''
test 3
001000.0
050005.0
010801.0
'''
Patient = '081'

data_Apop_081 = dataImport(Patient, 'Apoptosis')
data_Prol_081 = dataImport(Patient, 'Proliferation')

Pt81_Prol, Pt81_Apop = normalize(data_Prol_081, data_Apop_081)

ind1 = split(Pt81_Apop, Pt81_Prol, 0, 0)
ind2 = split(Pt81_Apop, Pt81_Prol, 0.5, 0.5)
ind3 = split(Pt81_Apop, Pt81_Prol, 0.1, 0.1)

plot(Patient, Pt81_Prol, Pt81_Apop, ind1, ind2, ind3)

p81_1 = len(ind1)/len(Pt81_Prol)
p81_2 = len(ind2)/len(Pt81_Prol)
p81_3 = len(ind3)/len(Pt81_Prol)


Patient = '084'

data_Apop_084 = dataImport(Patient, 'Apoptosis')
data_Prol_084 = dataImport(Patient, 'Proliferation')

Pt84_Prol, Pt84_Apop = normalize(data_Prol_084, data_Apop_084)

ind1 = split(Pt84_Apop, Pt84_Prol, 0, 0)
ind2 = split(Pt84_Apop, Pt84_Prol, 0.5, 0.5)
ind3 = split(Pt84_Apop, Pt84_Prol, 0.1, 0.1)

plot(Patient, Pt84_Prol, Pt84_Apop, ind1, ind2, ind3)

p84_1 = len(ind1)/len(Pt84_Prol)
p84_2 = len(ind2)/len(Pt84_Prol)
p84_3 = len(ind3)/len(Pt84_Prol)








