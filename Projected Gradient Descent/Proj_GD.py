# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:46:37 2019

@author: ohtt
"""

# Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = pd.read_csv("A_mat.csv")
b = pd.read_csv("b_vec.csv")

A = A.as_matrix()
b = b.as_matrix()

print(A.shape,b.shape)
A = A.T

#Functions

def least_squares(A, b, x):
    return(np.linalg.norm(A.T.dot(x)-b)**2)

def least_squares_gradient(A, b, x):
    return(2*np.matmul(A,(np.matmul(A.T,x)-b)))
    
def projected_gradient_descent(init, steps, grad, proj,tau):
    xs = [init]
    i = 0
    listf = []
    L = 0
    for step in steps:
        i+=1
        print(i)
        G = grad(A,b,xs[-1])
        x = proj(tau,xs[-1] - step * G)
        xs.append(x)
        if step > 0:
            fx = least_squares(A,b,x)
            listf.append(fx)
            L = max(L,((np.linalg.norm(xs[i]-xs[i-1])/np.linalg.norm(G[i]-G[i-1]))))
    print("L : " + str(L))
    return xs,listf

def proj(tau,x):
    if np.sum(x) > tau:
        x = tau*x/np.linalg.norm(x)
        
    return(np.abs(x))

#Solve for CX:

x0 = np.random.uniform(0,10,(A.shape[0],1))
tau = 0.05
CX,listf = projected_gradient_descent(x0, [0.0000001]*20, least_squares_gradient, proj,tau)
#From CX to X:

EX = np.array(CX).reshape((21,16384))
DX = EX[-1]
X = DX.reshape((128,128))
plt.imshow(X.T)
plt.colorbar()
plt.show()

plt.plot(range(20),listf)
plt.show()