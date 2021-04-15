# In[1]: 
# packages
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# In[2]: 

def dataImport(Pattern, keys, rep, timeP):
    
    newKeys = []
    for i in timeP:
        newKeys = newKeys + [k+'.'+str(i) for k in ['000901','001000','010702','010801','010900','020503','020602','020701',
        '030304','030403','030502','040105','040204','040303','050005','050104']]
    
    data = pd.DataFrame(columns=newKeys)
    
    for i in range(rep):
        mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, i)).item())
        
        temp = pd.Series()
        for j in timeP:
            temp = temp.append(mat[keys].iloc[j])
        
        data = data.append(pd.DataFrame([temp.values],columns=newKeys), ignore_index=True)
                
    return data


def dataMerge(keys, rep, timeP):
    
    dataE = dataImport('Exponential', keys, rep, timeP)
    dataL = dataImport('Logistic', keys, rep, timeP)
    dataH = dataImport('Hill', keys, rep, timeP)

    data = pd.concat([dataE, dataL, dataH])
    data = data.set_index(np.arange(3*rep))
    
    return data


def dataProportion(data):
    
    n = len(data)
    
    data[data < 1] = 0
    
    for i in range(n):
        data.loc[i] = data.loc[i]/sum(data.loc[i])
        
    return data


def labelGet(n=300):
    
    l1 = [0]*n
    l2 = [1]*n
    l3 = [2]*n
    
    return np.array(l1+l2+l3)


def feat_importance(model, data, label, n=5, seed=2021):
    
    importances = 0

    skf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)
    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        model.fit(X_train, y_train)
        results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=2)
        importances += results.importances_mean
        
    return importances/n


def accumulated_importance(importance):
    
    accu_importance = []
    
    for i in range(len(importance)):
        
        temp = sum(importance[:i+1])/sum(importance)
        accu_importance.append(temp)
    
    return accu_importance



def plotImportance(feature, importance):
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'30'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'30'}
    
    fig = plt.figure(figsize=(15,10))
    
    ax1 = fig.add_subplot(111)
    sorted_idx = importance.argsort()[::-1]
    sorted_feature = [feature[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    ax1.bar(sorted_feature, sorted_importance, color='cornflowerblue', label='Feature importance') 
    ax1.axhline(y=0, lw=1, color='k')
    ax1.tick_params(length=10, labelsize=25)
    plt.xticks(rotation=70)
    plt.legend(loc=(0.4, 0.6), prop=font2, frameon=False)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    ax2 = ax1.twinx()
    accu_importance = accumulated_importance(sorted_importance)
    ax2.plot(sorted_feature, accu_importance, lw=10, ls='-.', label='Accumulated importance', color='palevioletred')
    ax2.tick_params(length=10, labelsize=25)
    plt.ylim([-0.06, 1.1])
    ax1.set_xlabel('Subclones(D=Death, R=Resting, P=Proliferation)', font1)
    plt.legend(loc=(0.4, 0.5), prop=font2, frameon=False)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    plt.savefig('Feature_importance.tif', bbox_inches='tight')
    plt.savefig('Feature_importance.eps', bbox_inches='tight')
    plt.show()


# In[3]: 

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

X = dataMerge(keys, 300, [0])
X = dataProportion(X)
y = labelGet(300)
## random forest
rf = RandomForestClassifier(n_estimators=100, random_state=2021)
importance = feat_importance(rf, X, y, n=5, seed=2021)
plotImportance(keys, importance)
















