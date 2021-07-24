# In[1]: 
# packages
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

# In[2]: 
## ode model
def dataImport(Pattern, keys, rep):
    
    newKeys = ['000901','001000','010702','010801','010900','020503','020602','020701',
               '030304','030403','030502','040105','040204','040303','050005','050104']
    
    data = pd.DataFrame(columns=newKeys)
    
    for i in range(rep):
        mat = pd.DataFrame(np.load('%s\clone_%s_%d.npy'%(Pattern, Pattern, i)).item())
        
        temp = pd.Series()
        temp = temp.append(mat[keys].iloc[0])
        
        data = data.append(pd.DataFrame([temp.values],columns=newKeys), ignore_index=True)
                
    return data


def dataMerge(keys, rep):
    
    dataE = dataImport('Exponential', keys, rep)
    dataL = dataImport('Logistic', keys, rep)
    dataH = dataImport('Hill', keys, rep)

    data = pd.concat([dataE, dataL, dataH])
    data = data.set_index(np.arange(3*rep))
    
    return data


def dataProportion(data):
    
    n = len(data)
    
    data[data < 1] = 0
    
    for i in range(n):
        data.loc[i] = data.loc[i]/sum(data.loc[i])
        
    return data


def labelGet(n=200):
    
    l1 = [0]*n
    l2 = [1]*n
    l3 = [2]*n
    
    return np.array(l1+l2+l3)


def plot():
    dot_data = export_graphviz(dtree, out_file=None,
                               feature_names=iris.feature_names,
                               class_names=iris.target_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename ="iris", directory ='./', format='pdf')

# In[3]: 
## accuracy of different classifier
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

X = dataMerge(keys, 200)
X = dataProportion(X)
X = X[['001000', '000901']]
y = labelGet(200)

dt = tree.DecisionTreeClassifier(max_depth=2)
dt = dt.fit(X, y)

tree.plot_tree(dt) 


# In[4]: 

X = dataMerge(keys, 300)
X = dataProportion(X)
X = X[['001000']]
y = labelGet(300)

dt = tree.DecisionTreeClassifier()
dt = dt.fit(X, y)
dt.feature_importances_
tree.plot_tree(dt) 















