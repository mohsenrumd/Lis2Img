#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:32:51 2021

@author: mohsenr
"""

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
confusion_pred = np.zeros((21, 4, 4))
confusion_lis = np.zeros((21, 4, 4))

classes = set(chorals)

acc_pred = np.zeros((21,))
acc_lis = np.zeros((21,))

for sbj in range(21):
    data_pred =  load.unpickle(f'../Pred_img/sbj{sbj}_Pred.pkl')
    data_img = eeg[sbj,-44:,:,:1394]
    data_lis = eeg[sbj,:44,:,:1394]
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                    LinearModel(LogisticRegression(penalty = 'l1', C=.5, 
                                solver = 'liblinear')))
    
    
    
    decoder = SlidingEstimator(clf,scoring='roc_auc',n_jobs=-1,verbose=True)
    # Compute confusion matrix for each cross-validation fold
    decoder.fit(X = data_pred, y = chorals[idx])
    y_pred = decoder.predict_proba(X = data_img)
    tmp = y_pred.mean(axis = 1)
    acc_pred[sbj] = sum(tmp.argmax(1) == chorals)/44
    for ii, train_class in enumerate(classes):
        for jj in range(ii, len(classes)):
            confusion_pred[sbj, ii, jj] = roc_auc_score(chorals == train_class,tmp[:, jj])
            confusion_pred[sbj, jj, ii] = confusion_pred[sbj, ii, jj]
  
    labels = [f'Choral {i}' for i in range(1,5)]
    fig, ax = plt.subplots(1)
    im = ax.matshow(confusion_pred[sbj,:,:], cmap='RdBu_r', clim=[0.1, .9])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_title(f' PRED. to IMAG. Subject {sbj+1}_{round(acc_pred[sbj],2)}')
    clt = plt.colorbar(im)
    clt.set_label('AUC')
    plt.tight_layout()
    plt.show()
    
    
    
    decoder = SlidingEstimator(clf,scoring='roc_auc',n_jobs=-1,verbose=True)
    # Compute confusion matrix for each cross-validation fold
    decoder.fit(X = data_lis, y = chorals)
    y_pred = decoder.predict_proba(X = data_img)
    tmp = y_pred.mean(axis = 1)
    acc_lis[sbj] = sum(tmp.argmax(1) == chorals)/44
    for ii, train_class in enumerate(classes):
        for jj in range(ii, len(classes)):
            confusion_lis[sbj, ii, jj] = roc_auc_score(chorals == train_class,tmp[:, jj])
            confusion_lis[sbj, jj, ii] = confusion_lis[sbj, ii, jj]

    labels = [f'Choral {i}' for i in range(1,5)]
    fig, ax = plt.subplots(1)
    im = ax.matshow(confusion_lis[sbj,:,:], cmap='RdBu_r', clim=[0.1, .9])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_title(f' Lis. to img. Subject {sbj+1}_{round(acc_lis[sbj],2)}')
    clt = plt.colorbar(im)
    clt.set_label('AUC')
    plt.tight_layout()
    plt.show()
    

CM_lis = confusion_lis.mean(0)
CM_pred = confusion_pred.mean(0)
#%%

labels = [f'Chorale {i}' for i in range(1,5)]
fig, ax = plt.subplots(1)
im = ax.matshow(CM_lis, cmap='RdBu_r', clim=[0.1, .9])
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(labels)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(labels, rotation=0, ha='center')
ax.set_title(f'lis. to imag. Accuracy = {round(acc_lis.mean(),2)}')
clt = plt.colorbar(im)
clt.set_label('AUC')
plt.tight_layout()
# plt.savefig('../Figures/CM_trainLis_testImg.pdf', bbox_inches='tight')
plt.show()

labels = [f'Chorale {i}' for i in range(1,5)]
fig, ax = plt.subplots(1)
im = ax.matshow(CM_pred, cmap='RdBu_r', clim=[0.1, .9])
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(labels)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(labels, rotation=0, ha='center')
ax.set_title(f'pred. to imag. Accuracy = {round(acc_pred.mean(),2)}')
clt = plt.colorbar(im)
clt.set_label('AUC')
plt.tight_layout()
# plt.savefig('../Figures/CM_trainPred_testImg.pdf', bbox_inches='tight')
plt.show()
