# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:00:46 2021

@author: richa
"""

%%cython
import numpy as np
from numpy import log2
import math
from math import log,pi,ceil,floor
import matplotlib.pyplot as plt
import cmath
from cmath import exp
from numpy import ones, zeros, sin, cos, array, roll, sqrt
import random
from random import uniform
import gc
from scipy import stats
from numba import jit,njit, prange,vectorize
import joblib
from joblib import Parallel, delayed
gc.collect()

"""Hierarchical Nature of disorder in homogeneous walks are tuned with the help of this function"""
def index(g, N):
    g=abs(g-N)
    if g % 2 == 1:return 0;
    if g % (2 ** 20) == 0:return 20;
    if g % (2 ** 19) == 0:return 19;
    if g % (2 ** 18) == 0:return 18;
    if g % (2 ** 17) == 0:return 17;
    if g % (2 ** 16) == 0:return 16;
    if g % (2 ** 15) == 0:return 15;
    if g % (2 ** 14) == 0:return 14;
    if g % (2 ** 13) == 0:return 13;
    if g % (2 ** 12) == 0:return 12;
    if g % (2 ** 11) == 0:return 11;
    if g % (2 ** 10) == 0:return 10;
    if g % (2 ** 9) == 0:return 9;
    if g % (2 ** 8) == 0:return 8;
    if g % (2 ** 7) == 0:return 7;
    if g % (2 ** 6) == 0:return 6;
    if g % (2 ** 5) == 0:return 5;
    if g % (2 ** 4) == 0:return 4;
    if g % (2 ** 3) == 0:return 3;
    if g % (2 ** 2) == 0:return 2;
    if g % (2) == 0:return 1;


""" Rotation of Qubits with disorder introduced at each lattice points"""
def rotation_1(N,eps, w):
  q = [pow(eps,index(g,N))*0.25*pi for g in range(2*N+1) ]
  disorder =array([exp(1j * uniform(-w, w) * pi) for g in range(2*N+1)])
  SIN = array([a*b for a,b in zip(sin(q),disorder)])  
  SINM = array([-a*b for a,b in zip(sin(q),disorder)])
  COS =array([a*b for a,b in zip(cos(q),disorder)])
  return array([[SIN, COS], 
                  [COS, SINM]])


"""Evolution of the Walker, returns averaged Standard deviation output"""
def qw_split_avg(eps, w):
    
    N = 10_000 #Initializing no. of time-steps
    a = 1 / sqrt(2.0) #Initializing orientation of the spin 
    b = 1j / sqrt(2.0)# at theta=45 for homogeneous hadamard walk
    avg_disorder = zeros(21) 
    r1 = rotation_1(N,eps, w) 
    psi = np.zeros((2, 2 * N + 1), dtype=complex)
    psi[0,N] = a
    psi[1, N] = b
    std_dev = np.zeros(N + 1,dtype=float)
    positions = np.arange(-N,N+1)
    pow_2 = [pow(2, i) for i in range(1, 21)]
    j = 0
    
    #Evolving for N time steps
    for n in range(1, N + 1):
        
      psi[:,N-n:N+n+1] = np.einsum("ijk,jk->ik", r1[:,:,N-n:N+n+1], psi[:,N-n:N+n+1], optimize="optimal")  # rotation theta1
      psi[0] = roll(psi[0], 1)  # shift up
      psi[1] = roll(psi[1], -1)  # shift down
      
      #Recording Std for powers of 2
      if n == pow_2[j]:
        psi_sq = abs(psi[0, N - n : N + n + 1]) ** 2 + abs(psi[1, N - n : N + n + 1]) ** 2
        sum_0 = np.sum([(a**2)* b for a, b in zip(positions[N - n : N + n + 1], psi_sq)])
        sum_1 = np.sum([(a * b) ** 2 for a, b in zip(positions[N - n : N + n + 1], psi_sq)])
        std_dev[n] = sqrt(sum_0 - sum_1)
        j += 1
        avg_disorder[j] += std_dev[n]/50
    return avg_disorder

#Parallelizing averaging of disorder through joblib library
def qw_split(eps,w):
  with joblib.parallel_backend(backend="threading"):
      parallel = Parallel(verbose=5)
      standard_dev= np.sum(parallel([delayed(qw_split_avg)(eps, w) for k in range(50)]),axis=0)
      return standard_dev

def main():
    #W is the width of disorder which helps in distinguishing the nature of walk
    W = [pi,0.2*pi,0.1*pi, 0.05*pi]

    #Parallelizing processing for different W parameters
    with joblib.parallel_backend(backend="threading"):
      parallel = Parallel(verbose=5)
      standard_dev= (parallel([delayed(qw_split)(1, j) for j in W]))
    print(standard_dev)     
    
    
    for j in range(2):
      log_std=[]
      n=0
      for i in standard_dev[j]:

        if i != 0:
          n+=1
          log_std.append(log(i, 2)/n)
        if i==0:
          log_std.append(0)

  
      log_scale =[1/i for i in range(1,14)]
      log_scale =[0]+log_scale        
      res = np.polyfit(log_scale[10:14] ,log_std[10:14],1)
      log_std[0] = res[1]
      print("Intercept for W = "+str(W[j])+" ="+str(res[1]))
      fig2 = plt.figure(2)
      ax2 = fig2.add_subplot(111)
      plot2 = plt.figure(2)
      ax2.plot(log_scale,log_std[0:14], "x", label="W= " + str(W[j]))
      ax2.legend(loc="lower right", frameon=False)
      plt.xlabel("Log-time steps")
      plt.ylabel("Log-Mean Squared Displacement")
      plt.title("STDEv for Hierarchical Disorder")
      plt.xlim(0, 0.2)
      plt.savefig(f"plots/regular_disorder_plots/{round(j,2)} plot.png", dpi=600)


main()
