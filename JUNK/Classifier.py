#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:33:49 2021

@author: mohsenr
"""

import numpy as np
from eelbrain import *
import mne
from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch

[a, _] = random_split(range(44), [44, 0], generator=torch.Generator().manual_seed(42))
idx = a.indices
chorals = np.asarray([i//11 for i in range(44)])
# chorals = chorals[idx]
confusion = np.zeros((21, 4, 4))
acc_pred = np.zeros((21,180))
acc_img = np.zeros((21,180))
acc_lis = np.zeros((21,180))

for sbj in range(21):
    data_pred =  load.unpickle(f'../Pred_img/sbj{sbj}_Pred.pkl')
    data_img = eeg[sbj,-44:,:,:1802]
    data_lis = eeg[sbj,:44,:,:1802]
    
    
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                    LinearModel(LogisticRegression(penalty = 'l2', C=1, 
                                solver = 'liblinear')))
    decoder = SlidingEstimator(clf,scoring='accuracy',n_jobs=-1,verbose=True)
    # score = cross_val_multiscore(decoder, X=data[:,:,:], y=chorals, n_jobs=-1, cv=5)
    # print(score.mean())
    
    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    # Compute confusion matrix for each cross-validation fold
    X = data_pred
    y = chorals[idx]
    y_pred = np.zeros((len(y), 1802,4))
    for train, test_t in cv.split(X, y):
        # Fit
        decoder.fit(X[train], y[train])
        # Probabilistic prediction (necessary for ROC-AUC scoring metric)
        y_pred[test_t] = decoder.predict_proba(X[test_t])
    for i in range(0,180):    
        tmp = y_pred[:,:(i+1)*10,:].mean(axis = 1)
        acc_pred[sbj,i] = sum(tmp.argmax(1) == y)/44
        
        
        
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                    LinearModel(LogisticRegression(penalty = 'l2', C=1, 
                                solver = 'liblinear')))
    decoder = SlidingEstimator(clf,scoring='accuracy',n_jobs=-1,verbose=True)  
    X = data_img
    y = chorals
    y_img = np.zeros((len(y), 1802,4))
    for train, test_t in cv.split(X, y):
        # Fit
        decoder.fit(X[train], y[train])
        # Probabilistic prediction (necessary for ROC-AUC scoring metric)
        y_img[test_t] = decoder.predict_proba(X[test_t])
    for i in range(0,180):    
        tmp = y_img[:,:(i+1)*10,:].mean(axis = 1)
        acc_img[sbj,i] = sum(tmp.argmax(1) == y)/44


    clf = make_pipeline(StandardScaler(),  # z-score normalization
                    LinearModel(LogisticRegression(penalty = 'l2', C=1, 
                                solver = 'liblinear')))
    decoder = SlidingEstimator(clf,scoring='accuracy',n_jobs=-1,verbose=True)  
    X = data_lis
    y = chorals
    y_lis = np.zeros((len(y), 1802,4))
    for train, test_t in cv.split(X, y):
        # Fit
        decoder.fit(X[train], y[train])
        # Probabilistic prediction (necessary for ROC-AUC scoring metric)
        y_lis[test_t] = decoder.predict_proba(X[test_t])
    for i in range(0,180):    
        tmp = y_lis[:,:(i+1)*10,:].mean(axis = 1)
        acc_lis[sbj,i] = sum(tmp.argmax(1) == y)/44
#%%
acc_pred_mn = acc_pred.mean(0)
acc_pred_sem = acc_pred.std(0)/np.sqrt(21)
acc_img_mn = acc_img.mean(0)
acc_img_sem = acc_img.std(0)/np.sqrt(21)
acc_lis_mn = acc_lis.mean(0)
acc_lis_sem = acc_lis.std(0)/np.sqrt(21)
t = np.linspace(0,28,180)
p_05_pred = np.ma.masked_where(p_val_pred>0.05, acc_pred_mn)
p_05_img = np.ma.masked_where(p_val_img>0.05, acc_img_mn)
p_05_lis = np.ma.masked_where(p_val_lis>0.05, acc_lis_mn)

fig, ax = plt.subplots()
ax.plot(t, acc_pred_mn, label='Predicted imagined', lw = .5)
ax.plot(t, acc_img_mn, label='True imagined', lw = .5)
ax.plot(t, acc_lis_mn, label='True listened', lw = .5)

ax.plot(t, p_05_pred, c = [0., 0.4, .6], lw = 2)
ax.plot(t, p_05_img, c = [0.8, 0.3, .1], lw = 2)
ax.plot(t, p_05_lis, c = [0.3, 0.7, .1], lw = 2)

ax.fill_between(t, acc_pred_mn-acc_pred_sem, acc_pred_mn+acc_pred_sem, alpha = 0.3) 
ax.fill_between(t, acc_mn_true-acc_sem_true, acc_mn_true+acc_sem_true, alpha = 0.3) 
ax.fill_between(t, acc_lis_mn-acc_lis_sem, acc_lis_mn+acc_lis_sem, alpha = 0.3) 
ax.axhline(.25, color='k', linestyle='--', label='chance')
ax.set_xlabel('time(s)')
ax.set_ylabel('Accuracy')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title(f'Accuracy over time')
ax.set_ylim(.2,.72)
# plt.savefig(f'../Figures/Over_time_All.pdf', bbox_inches='tight')
