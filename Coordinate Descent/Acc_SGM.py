# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:47:19 2019

@author: ohtt
"""

import numpy as np
import matplotlib.pyplot as plt

def A(N):
    P = np.random.uniform(size = (N,N))
    #Create Q
    Q, _ = np.linalg.qr(P,mode = 'complete')
    
    #Create D
    xi = np.random.uniform(size = N)
    
    D = np.diag(10**(-xi))
    #print(D)
    
    return(np.matmul(Q.T,np.matmul(D,Q)))

def f(x,A,b):
    return(0.5*np.dot(x.T,np.dot(A,x)) - np.dot(b,x))
    
def gradnoise(x,A,b,sigma,eps):
    return(np.matmul(A,x)-b + sigma*eps)
def grad(x,A,b):
    return(np.matmul(A,x)-b)
    
N = 100
eps = np.random.normal(loc = 0, scale = 1,size = N)
sigma = 0.1
x0 = np.random.uniform(size = N)
b = np.random.uniform(size = N)
step = 0.1
A = A(N)
eig,_ = np.linalg.eig(A)
maxeig = np.max(eig)
error1 = []
error2 = []
error3 = []
error4 = []
iter = 300
beta = 0.9 

if maxeig < 4:
    #Gradient Descent Noisy
    x = x0
    for i in range(iter):
        x = x - step*gradnoise(x,A,b,sigma,eps)
        
        error1.append(f(x,A,b))
    
    #Nesterov Noisy
    x, xprec = x0, x0
    for i in range(iter):
        y = x + beta * (x - xprec)
        xprec = x
        x = y - step*gradnoise(y,A,b,sigma,eps)
        error2.append(f(x,A,b))
       
    #Gradient Descent
    x = x0
    for i in range(iter):
        x = x - step*grad(x,A,b)
        error3.append(f(x,A,b))

    #Nesterov
    x, xprec = x0, x0
    for i in range(iter):
        y = x + beta * (x - xprec)
        xprec = x
        x = y - step*grad(y,A,b)
        error4.append(f(x,A,b))
    
plt.plot(range(iter),error1,label = "Gradient Descent Noisy")
plt.plot(range(iter),error2,label = "Nesterov Noisy")
plt.plot(range(iter),error3,label = "Gradient Descent")
plt.plot(range(iter),error4,label = "Nesterov")
plt.legend()
plt.show()
