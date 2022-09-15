# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:06:18 2018

@author: Varun
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
r = np.random.normal(0,1,10000)
b = np.histogram(r,40)
bins = (b[1][1:]+b[1][:-1] )/ 2
plt.plot(bins,b[0])
np.mean(r)

r1 =  r[:5000]
r2 = -r1
rh = np.concatenate([r1,r2])
bh = np.histogram(rh,40)
binsh = (bh[1][1:]+bh[1][:-1] )/ 2
plt.plot(binsh,bh[0])

np.mean(rh)