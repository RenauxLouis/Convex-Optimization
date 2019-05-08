# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:47:19 2019

@author: ohtt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Adult.csv")

data = data.as_matrix()
print(data.shape)

X = np.array((len(data),10))
Y = np.array((len(data),1))

X_test = data[:8000,0:10]
Y_test = data[:8000,10]
X_train = data[:,0:10]
Y_train = data[:,10]

N = 10000
lambda1 = 0.001
alpha = 0.5
beta =  0.9
b = 1

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def Opti(OX, OY, ON, Oalpha,Ob):
        
    wminus1 = np.zeros(10)
    bminus1 = 0
    w = np.zeros(10)
    b = 1
    m = X_train.shape[0]
    lista = []
    
    for step in range(ON):
        Oalpha = Oalpha**(1)
        z = b + np.dot(OX, w)
        h = sigmoid(z)
        
        dw = (1/m)*(np.dot(OX.T, (h-OY.T).T)) + lambda1*2*w
        db = (1/m)*(np.sum(h-OY.T))
            
        #Momentum Gradient Method:
        
        wminus1, w = w, w - (Oalpha * (dw.T)) + beta*(w-wminus1)
        bminus1, b = b, b - (Oalpha * db) + beta*(b-bminus1)
        
        #Gradient Method:
        
        #w = w - (Oalpha * (dw.T))
        #b = b - (Oalpha * db)
        
        a = f(X_train,Y_train,lambda1,w,b)
        lista.append(a)
        
        if a < 12116:
            plt.plot(range(step+1),lista)
            plt.show()
            laststep = step
            break
        
    return w,b,laststep
    
def f(fX,fY,flambda1,fw,fb):
    s = fb + np.dot(fX, fw)
    fsum = np.sum(-fY*s + np.log(1 + np.exp(s)))
    fnorm = flambda1*np.linalg.norm(fw)**2
    return(fsum + fnorm)
    
w,b,lestep = Opti(X_train, Y_train, N, alpha,b)
print(lestep)
print(f(X_train,Y_train,lambda1,w,b))





