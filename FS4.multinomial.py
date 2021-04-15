import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt


def Probability(p1, p2, n1, n2, N):
    
    Prob = math.factorial(N)/math.factorial(n1)/math.factorial(n2)/math.factorial(N-n1-n2)*p1**n1*p2**n2*(1-p1-p2)**(N-n1-n2)
    
    return Prob


def distribution(p1, p2, N):
    
    dic = {}
    
    for i in range(N+1):
        for j in range(0,N+1-i):
            cellNum = 1*i + 2*j
            Prob = Probability(p1, p2, i, j, N)
            temp = {cellNum: Prob}
            dic = dict(Counter(dic)+Counter(temp))   # Merge the dictionary
    
    return dic


def normfun(x,mu,sigma):
    
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    
    return pdf


def barplot(p1, p2, s, N=100):
    
    dic = distribution(p1, p2, N)
    Num = list(dic.keys())
    p = list(dic.values())

    fig, ax = plt.subplots(figsize=(13,11))
    
    x = np.arange(65, 140, 0.1)
    y = normfun(x, N, np.sqrt(N*2*p2))
    plt.plot(x,y, color='thistle', lw=6)
    
    ax.bar(Num, p, width=2, color='cornflowerblue', label='(%.1f, %.1f, %.1f)'%(1-p1-p2, p1, p2))
    
    font1 = {'family':'Arial', 'weight':'normal', 'size':'50'}
    font2 = {'family':'Arial', 'weight':'bold', 'size':'30'}

    ax.set_xlabel('Cell number', font1)
    ax.set_ylabel('Probability', font1)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlim(65, 140)

    ax.legend(loc='upper right', prop=font2, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('Figure_7_%s.tif'%s, bbox_inches='tight')
    plt.savefig('Figure_7_%s.eps'%s, bbox_inches='tight')
    plt.show()
    
    
barplot(1, 0, 'a', N=100)
barplot(0.8, 0.1, 'b', N=100)
barplot(0.6, 0.2, 'c', N=100)
barplot(0.4, 0.3, 'd', N=100)
barplot(0.2, 0.4, 'e', N=100)
barplot(0.02, 0.49, 'f', N=100)
















