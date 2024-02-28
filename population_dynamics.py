import numpy as np
import pandas as pd
from scipy.integrate import odeint


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


para1 = [0.01, 0.008]
para2 = [0.01, 500000000000, 0.008]
para3 = [0.01, 1, 0.00000000001, 0.008]

CellNumE = CellNumber(3000, Exponential, para1)[:,0]
CellNumL = CellNumber(3000, Logistic, para2)[:,0]
CellNumH = CellNumber(3000, Hill, para3)[:,0]






