# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:20:34 2018
This python code uses different forms of LDA after detecting change points to separate mixture of two time series data
for example:
    ground truth - 
    series 1 = R R G R R G R R
    series 2 =          B   G   B   G   B
    What do we see?
    R R G R RBG RGR  B   G   B
    Question:
        how many series form the mixture?
        what is the distribution (time interval and color distribution) of each series?
        what is the start and end time of each series?
        
Change point definition - 
If we have a sequence X_1,..,X_n
and X_1,..,X_{ch - 1} are from one distribution and X_{ch},...,X_{n} are from another
then we say that ch is the change point. (point where the new distribution starts)
@author: ssatpth2
"""
import numpy as np
import pandas as pd
import scipy.stats
import time
import matplotlib.pyplot as plt
import lda
import gensim
def getcumcount(tseries,window):
    if window == 0:
        X = pd.get_dummies(tseries)
        return X, X.cumsum(), X.iloc[::-1].cumsum()[::-1]
def computeDist(phat,qhat,n,m,metric):
    if metric == 'L1':
        return (abs(phat - qhat)).sum(axis = 1)
    if metric == 'L1someother':
        return   ( ( (phat - qhat)**2 - phat.div(n,axis =0 ) - qhat.div(m,axis = 0) ) /  (phat + qhat) ).sum(axis = 1)
    if metric == 'L2plugin':
        return np.sqrt(((phat - qhat)**2).sum(axis =1))
    if metric == 'L2unbiased':
        pphat = (phat**2 - phat.div(n,axis = 0)).sum(axis =1) / (1 - 1/n)
        qqhat = (qhat**2 - qhat.div(m,axis = 0)).sum(axis = 1) / (1 - 1/m)
        pqhat = (phat*qhat).sum(axis = 1)
        return np.sqrt(np.maximum(pphat  + qqhat - 2*pqhat,0))    
    if metric == 'JS':
        mhat = (phat+qhat)/2
        return pd.Series(np.sqrt(scipy.stats.entropy(phat.T,mhat.T)/2 + scipy.stats.entropy(qhat.T,mhat.T)/2),index = phat.index)
    if metric == 'hellinger':
        return np.sqrt(((np.sqrt(phat) - np.sqrt(qhat))**2).sum(axis = 1))/np.sqrt(2)
    if metric == 'Bhattacharya':
        return -np.log(np.sqrt(phat*qhat).sum(axis = 1))  
def estimateChange(st,end,alpha,thresh,metric):
    #there needs to be atleast three points to have a unimodal function (function with a peak)
    if len(range(int( alpha*(end - st + 1) ) , int( (1 - alpha)*(end - st + 1) ))) < 3:
        return 'nochpt',''
    L = Xcum.iloc[st:end+1] - Xcum.iloc[st]
    R = Xcuminv.iloc[st:end+1] - (Xcuminv.iloc[end +1] if end <len(Xcum) -1 else 0) 
    phat = L.div(L.sum(axis = 1),axis = 0)#distribution of left
    qhat = R.div(R.sum(axis = 1),axis = 0)#distribution of right
    N = L.sum(axis = 1)#number of points in left
    M = R.sum(axis = 1)#number of points in right
    distance = computeDist(phat,qhat,N,M,metric)
    chptTime = distance.iloc[int( alpha*(end - st + 1) ) : int( (1 - alpha)*(end - st + 1) )].idxmax()
    if pd.isnull(chptTime):
        return 'nochpt',''
    try:
        chptind = distance.index.get_loc(chptTime).start
    except AttributeError:
        chptind = distance.index.get_loc(chptTime)
    maxval = distance.iloc[chptind]
    if maxval > thresh and chptTime != distance.index[int( alpha*(end - st + 1) )] and chptTime != distance.index[int((1 - alpha)*(end - st + 1)) - 1 ]:
        return chptTime,st + chptind
    else:
        return 'nochpt',''
def CD(st,end,alpha,thresh,metric):
    chptTime,chptind = estimateChange(st,end,alpha,thresh,metric)
    if chptTime != 'nochpt':          
        chptLeft,chptLeftind = CD(st,chptind - 1,alpha,thresh,metric)
        chptRight,chptRightind = CD(chptind,end,alpha,thresh,metric)
        return chptLeft + [chptTime] + chptRight, chptLeftind + [chptind] + chptRightind
    else:
        return [],[]   
def LDA(changepointsTime,changepointsInd, alg = 'GibbsLDA'):
    X['episode'] = pd.cut(list(range(0,len(X))),bins = [0] + changepointsInd + [np.inf],right = False, include_lowest = False, labels = list(zip(  [X.index[0]] + changepointsTime ,  list(X.index[np.array(changepointsInd) - 1]) + [X.index[-1]]) )  )    
    XLDA = X.groupby('episode').sum()
    vocab = list(XLDA.columns)
    if alg == 'GibbsLDA':
        model = lda.LDA(n_topics=2, n_iter=500, random_state=1)
        model.fit(np.array(XLDA.astype(int)))
        topic_word = model.topic_word_
        n_top_words = 4
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        return model, vocab, list(X.index)
    if alg == 'variationalOnline':
        dic = gensim.corpora.dictionary.Dictionary([vocab])
        corpus = gensim.matutils.Dense2Corpus(np.array(XLDA.T.astype(int)))
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=2,passes = 20,id2word = dic,update_every = 0,alpha = 'auto',eta = 'auto',minimum_probability = 0.01)
        return lda_model,vocab,list(X.index)
    if alg == 'spectralLDA':
        alpha, beta = spectral_lda(np.array(XLDA), 0.1, 2, l1_simplex_proj=True, n_partitions=1)
        return (alpha,beta),vocab,list(X.index)
    """
    if alg == 'ISLE':
        #for now this uses an exe build through a c code and calls through a file. This can be optimized if we create a python wrapper of the c code.
        f = open('tdf_file.txt','w')
        arr = np.array(XLDA)
        [ [f.write('%d %d %d\n' %(i,j,val)) for j,val in enumerate(row,1) if val>0] for i,row in enumerate(arr,1)]
        f.close()
        f1 = open('vocab_file.txt','w')
        [f1.write(word + '\n') for word in vocab]
        f1.close()
        vocab_size = ''
        subprocess.run('ISLETrain <tdf_file> <vocab_file> <output_dir> <vocab_size> <num_docs> <max_entries> <num_topics> <sample(0/1)> <sample_rate> <edge topics(0/1) <max_edge_topics>')
    """

startTime = time.time()
X, Xcum,Xcuminv = getcumcount(tseries,0)
alpha = 0.15
thresh = 0.3
metric = 'L1'
MIN_EPISODE_LENGTH = int(alpha*len(X))
st = 0;end = len(X) - 1
changepointsTime,changepointsInd = CD(st,end,alpha,thresh,metric)
print("Time Taken for CD: ",time.time() - startTime)
print("Change points found", changepointsInd)
print("true changepoints",chpt)
model,msgs,episodes = LDA(changepointsTime, changepointsInd, 'GibbsLDA')
print('event-message distribution')
##if you are using GibbsLDA use this print statement to print the event message distribution
print(model.topic_word_)
##if you are using variationalOnline method please use the statements below to print
#print(model.print_topics(num_topics=2, num_words=4)) #or model.get_topics()
##if you are using spectralLDA please use the statements below to print
#print(model[1])


"""
#this piece compares different versios of LDA
model,msgs,episodes = LDA(changepointsTime, changepointsInd, 'spectralLDA')
spectralLDA = model[1]
model,msgs,episodes = LDA(changepointsTime, changepointsInd, 'variationalOnline')
onlinevar = model.get_topics()
model,msgs,episodes = LDA(changepointsTime, changepointsInd, 'GibbsLDA')
gibbslda = model.topic_word_
print( 'spectral lda vs true dist', np.linalg.norm(truedist - spectralLDA.T,np.inf)) 
print('gibbs lda vs true dist', np.linalg.norm(truedist[[1,0],:] - gibbslda,np.inf))
print( 'online vartional inference vs true dist'  ,np.linalg.norm(truedist[[1,0],:] - onlinevar,np.inf) )
"""
