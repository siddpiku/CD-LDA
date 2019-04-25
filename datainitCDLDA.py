# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:03:21 2018

@author: ssatpth2
data initialization for CDLDA
"""
import numpy as np
import pandas as pd
import time

def dist(dim,which):
    if which == 'uniform':
        #one distribution is uniform, another is almost uniform with first half epsilon less than uniform and second half epsilon more.
        dim1 = int(np.floor(0.5*dim))
        epsilon = 0.01
        dist1 = np.ones(dim)/dim 
        dist1[:dim1] = dist1[:dim1] - epsilon
        dist1[dim1:] = dist1[dim1:] + epsilon     
        dist1 = dist1/dist1.sum()
        dist2 = np.ones(dim)/dim     
    if which == 'suppdiff':
        #the two distributinos are different by the difference in their support sets
        dim1 = int(np.floor(0.5*dim))
        dim2 = int(np.floor(0.25*dim))
        dist1 = 0.001 + np.hstack( ( 0.5*np.ones(dim1)/dim1, 0.5*np.ones(dim2)/dim2, np.zeros(dim - dim1-dim2)) )
        dist2 = 0.001 + np.hstack( ( 0.5*np.ones(dim1)/dim1, np.zeros(dim2), 0.5*np.ones(dim - dim1-dim2)/(dim - dim1-dim2) ) )
        dist1 = dist1/dist1.sum()
        dist2 = dist2/dist2.sum()        
    return dist1,dist2

def datainit(dim,n):
    overlap = 0.3#if overlap is c then second series stats at (1 -c)n/2 
    dist1,dist2 = dist(dim,'suppdiff')
    timedata = np.random.poisson(lam = (2,3),size = (int(n/2),2)).cumsum(axis = 0)
    offset = timedata[int(n/2-1) - int(overlap*n/2),0]
    valuedata1 = np.random.choice(dim,int(n/2),p = list(dist1))
    valuedata2 = np.random.choice(dim,int(n/2),p = list(dist2))
    pdseries1 = pd.DataFrame({'time': pd.to_datetime(timedata[:,0],unit = 's'),'val': valuedata1})
    pdseries2 = pd.DataFrame({'time': pd.to_datetime(offset+timedata[:,1],unit = 's'),'val': valuedata2})
    data = pd.merge_ordered(pdseries1,pdseries2,on = 'time').set_index('time').stack().reset_index(level =1).drop(['level_1'],axis = 1).rename(index = str, columns = {0:'msg'})
    chpt2 = data.index.get_loc( str(pd.to_datetime([timedata[-1,0]],unit = 's')[0]) )
    try:
        chpt2 = chpt2 / (n*1.0)
    except TypeError:
        chpt2 = chpt2.start /  (n*1.0)  
    chpt = [(1-overlap)/2, chpt2] 
    return dist1,dist2,chpt,data.astype(int).astype(str)
dim = 4
n = int(1e4)
st = time.time()
dist1,dist2,chpt,tseries = datainit(dim,n) 
print(time.time()-st)
