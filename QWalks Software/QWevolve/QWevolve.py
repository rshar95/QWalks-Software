# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:27:28 2021

@author: richa
"""

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



def rotation_1(N,eps):
  q = [pow(eps,index(g,N))*0.25*pi for g in range(2*N+1) ]
  return array([[sin(q), cos(q)], 
                  [cos(q), -sin(q)]])



def qw_split(eps,N):
    
    a = 1 / sqrt(2.0)
    b = 1j / sqrt(2.0)
    avg_disorder = zeros(21)
    r1 = rotation_1(N,eps)
    psi = np.zeros((2, 2 * N + 1), dtype=complex)
    psi_t = zeros((2, 2*N + 1, N+1), dtype = complex)
    psi_t[:,:,0] = psi
    psi[0,N] = a
    psi[1, N] = b

    for n in range(1, N + 1):
        
      psi[:,N-n:N+n+1] = np.einsum("ijk,jk->ik", r1[:,:,N-n:N+n+1], psi[:,N-n:N+n+1], optimize="optimal")  # rotation theta1
      psi[0] = roll(psi[0], 1)  # shift up
      psi[1] = roll(psi[1], -1)  # shift down
      psi_t[:,:,n] = psi
    return psi_t

def measure(psi):
    return abs(psi[0,:])**2 + abs(psi[1,:])**2


def main():
    N=1000
    P=2*N+1
    eps=0.7
    for n in range(0,N+1,100):
      psi_t=qw_split(eps,N)
      psi = psi_t[:,:,n]
      prob = measure(psi)
      fig1 = plt.figure(1)
      ax1 = fig1.add_subplot(111)
      plt.title("Espilon= "+str(eps)+"  n= "+str(n))
      plot1=plt.figure(1)
      ax1.plot(range(P), prob)
      ax1.plot(range(P), prob, 'o')
      loc = range (0, P, int(P / 10)) #Location of ticks
      plt.xticks(loc)
      plt.xlim(0, P)
      ax1.set_xticklabels(range (-N, N+1,int(P/10)))
      plt.show()   
      plt.savefig(f"plots/QWevolve/{round(j,2)} plot.png", dpi=600)


main()
