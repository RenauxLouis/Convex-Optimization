# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:50:41 2019

@author: ohtt
"""

import numpy as np
import random as rd
  
def f(x):
    return(0.5*np.dot(x.T,np.dot(A,x)))
    
def gradf(x):
    return(np.dot(A[i,:],x))
    
def fullgradf(x):
    return(np.matmul(A,x))
    
def fA(N,test):
    #Create P
    P = np.random.uniform(size = (N,N))
    
    #Create Q
    Q, _ = np.linalg.qr(P,mode = 'complete')

    #Create D
    if test == "1" or test == "3":
        xi = np.random.uniform(size = N)
    elif test == "2":
        xi = np.random.uniform(low=0.0,high=2.0,size = N)
    
    D = np.diag(10**(-xi))
    eig,_ = np.linalg.eig(D)
    mu = np.min(eig)
    
    if test == "3":
        return(np.matmul(Q.T,np.matmul(D,Q))+5*np.ones((N,N)),mu)
    else:
        return(np.matmul(Q.T,np.matmul(D,Q)),mu)


def CD(x,A,eps,methodstep, method,L,Lmax,mu):
    global a1,a2
    print("Method " + method + " with " + methodstep)
    i = 0
    listi = [i]
    n = 0
    x0 = x
    a1 = 1
    a2 = 1

    if methodstep == "L":
        step = 1/L
    elif methodstep == "Lmax":
        step = 1/Lmax
    elif methodstep == "nL":
        step = 1/(np.sqrt(N)*L)

    while f(x) > eps*f(x0):
        
        if method == "Stochastic":
            i = rd.randrange(N)
        elif method == "Cyclic":
            i = (i+1)%N
        elif method == "Sampling":
            if i == listi[-1]:
            	listi = list(range(N))
            	np.random.shuffle(listi)
             
            else:
                listi = listi[1:]
            i = listi[0]
        
        direc = np.zeros(N)
        direc[i] =  np.dot(A[i,:],x)

        if methodstep == "exact":
            step = (np.matmul(x.T,np.matmul(A,fullgradf(x))))/(np.matmul(fullgradf(x).T,np.matmul(A,fullgradf(x))))
        
        x = x - step*direc

        if method == "Stochastic" and methodstep == "Lmax":
        	gauche = f(x)
        	droite = ((1-(mu)/(N*Lmax))**n)*(f(x0))
        	if gauche > droite:
        		a1 = 0
        elif method == "Cyclic" and methodstep == "Lmax":
        	gauche = f(x)
        	alpha = 1/Lmax
        	droite = ((1-(mu)/((2/alpha)*n*(1+(Lmax**2)*alpha**2)))**n)*(f(x0))
        	if gauche > droite:
        		a2 = 0

        n += 1
        if n%(100*N) == 0:
        	print(n)
    return(f(x),x,n)


######### Test 1

N = 100
eps = 0.000001
x0 = np.random.uniform(size = N)
L = 0
Lmax = 0

for l in ["1","2","3"]:
    A,mu = fA(N,l)
    L = np.linalg.norm(A)
    Lmax = 0
    for i in range(N):
        Lmax = max(Lmax,A[i,i])
    print("A"+str(l))
    print(L,Lmax)
    
    fmin = {}
    xmin = {}
    n = {}
    
    for j in ["Stochastic","Sampling"]:
        fmin[j+"exact"], xmin[j+"exact"], n[j+"exact"] = CD(x0,A,eps,"exact", j,L,Lmax,mu)
        fmin[j+"Lmax"], xmin[j+"Lmax"], n[j+"Lmax"] = CD(x0,A,eps,"Lmax", j,L,Lmax,mu)
    
    fmin["Cyclic"+"exact"], xmin["Cyclic"+"exact"], n["Cyclic"+"exact"] = CD(x0,A,eps,"exact", "Cyclic",L,Lmax,mu)
    fmin["Cyclic"+"Lmax"], xmin["Cyclic"+"Lmax"], n["Cyclic"+"Lmax"] = CD(x0,A,eps,"Lmax", "Cyclic",L,Lmax,mu)
    fmin["Cyclic"+"L"], xmin["Cyclic"+"L"], n["Cyclic"+"L"] = CD(x0,A,eps,"L", "Cyclic",L,Lmax,mu)
    fmin["Cyclic"+"nL"], xmin["Cyclic"+"nL"], n["Cyclic"+"nL"] = CD(x0,A,eps,"nL", "Cyclic",L,Lmax,mu)
    
    print("A"+str(l))
    print("L = " + str(L),"Lmax = " + str(Lmax))
    
    if a1 == 1:
        print("Convergence expression of Theorem 6.1 valid")
    if a2 == 1:
        print("Convergence expression of Theorem 6.2 valid")
    
    for key,val in fmin.items():
    	print(key, "=>", val)
    for key,val in n.items():
    	print(key, "=>", val)
