#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:57:03 2021

@author: mohsenr

"""

import numpy as np
from eelbrain import *
import mne
from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch
from scipy.signal import resample

[a, _] = random_split(range(44), [44, 0], generator=torch.Generator().manual_seed(42))
idx = a.indices
chorals = np.asarray([i//11 for i in range(44)])
# chorals = chorals[idx]
confusion_pred = np.zeros((21, 4, 4))
confusion_lis = np.zeros((21, 4, 4))

classes = set(chorals)

acc_pred = np.zeros((21,180, 180))
acc_lis = np.zeros((21,180, 180))

for sbj in range(21):
    data_pred =  load.unpickle(f'../Pred_img/sbj{sbj}_Pred.pkl')
    data_pred = resample(data_pred[:,:,1:1801], 180, axis = 2)
    data_img = resample(eeg[sbj,-44:,:,:1802], 180, axis = 2)
    data_lis = resample(eeg[sbj,:44,:,:1802], 180, axis = 2)
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                    LinearModel(LogisticRegression(penalty = 'l1', C=1, 
                                solver = 'liblinear')))
    
    
    
    decoder = GeneralizingEstimator(clf,scoring='accuracy',n_jobs=-1,verbose=True)
    # Compute confusion matrix for each cross-validation fold
    decoder.fit(X = data_pred, y = chorals[idx])
    acc_pred[sbj,:,:] = decoder.score(X = data_img, y = chorals)

  
    fig, ax = plt.subplots(1)
    im = ax.matshow(acc_pred[sbj,:,:], vmin=.0, vmax=.5,
                    cmap='RdBu_r', origin='lower',)# extent = [0, 100, 0, 100])
    # ax.contour(p_val, levels = [-0.1, 0.05], colors = 'k', linestyles = '--', 
    #              linewidths =.35)
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(f'Pre to img: sbject{sbj+1}')
    clb = plt.colorbar(im)
    clb.set_label('Accuracy  (chance = 0.25)')
    plt.tight_layout()
    plt.show()
    
    
    decoder = GeneralizingEstimator(clf,scoring='accuracy',n_jobs=-1,verbose=True)
    # Compute confusion matrix for each cross-validation fold
    decoder.fit(X = data_lis, y = chorals)
    acc_lis[sbj,:,:] = decoder.score(X = data_img, y = chorals)

    fig, ax = plt.subplots(1)
    im = ax.matshow(acc_lis[sbj,:,:], vmin=.0, vmax=.5,
                    cmap='RdBu_r', origin='lower',)# extent = [0, 100, 0, 100])
    # ax.contour(p_val, levels = [-0.1, 0.05], colors = 'k', linestyles = '--', 
    #              linewidths =.35)
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(f'lis to img: sbject{sbj+1}')
    clb = plt.colorbar(im)
    clb.set_label('Accuracy  (chance = 0.25)')
    plt.tight_layout()
    plt.show()

#%%
def _stat_fun(x, sigma=0, method='relative'):
    from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
    """Aux. function of stats"""
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def stats(X, connectivity=None, n_jobs=-1):
    """Cluster statistics to control for multiple comparisons.
    Parameters
    ----------
    X : array, shape (n_samples, n_space, n_times)
        The data, chance is assumed to be 0.
    connectivity : None | array, shape (n_space, n_times)
        The connectivity matrix to apply cluster correction. If None uses
        neighboring cells of X.
    n_jobs : int
        The number of parallel processors.
    """
    from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p

    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=_stat_fun, n_permutations=2**12,
        n_jobs=n_jobs)
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T


def run_permutation_test(pooled,sizeZ,sizeY,delta):
    np.random.shuffle(pooled)
    starZ = pooled[:sizeZ]
    starY = pooled[-sizeY:]
    return starZ.mean() - starY.mean()


acc_mn_lis = acc_lis.mean(0)
p_value_lis = stats(acc_lis-.25)
acc_mn_pred = acc_pred.mean(0)
p_value_pred = stats(acc_pred-.25)
#%%

fig, ax = plt.subplots(1)
im = ax.matshow(acc_mn_lis, vmin=.1, vmax=.4,
                cmap='RdBu_r', origin='lower', extent = [0, 28, 0, 28])
ax.contour(p_value_lis, levels = [-0.1, 0.05], colors = 'k', linestyles = '--', 
              linewidths =.35, extent = [0, 28, 0, 28])
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title(f'lis to img')
clb = plt.colorbar(im)
clb.set_label('Accuracy  (chance = 0.25)')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(1)
im = ax.matshow(acc_mn_pred, vmin=.1, vmax=.4,
                cmap='RdBu_r', origin='lower', extent = [0, 28, 0, 28])
ax.contour(p_value_pred, levels = [-0.1, 0.05], colors = 'k', linestyles = '--', 
              linewidths =.35, extent = [0, 28, 0, 28])
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title(f'pred to img')
clb = plt.colorbar(im)
clb.set_label('Accuracy  (chance = 0.25)')
plt.tight_layout()
plt.show()