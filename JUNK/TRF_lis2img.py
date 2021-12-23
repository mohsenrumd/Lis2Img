#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:08:06 2021

@author: mohsenr
"""
import torch
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from eelbrain import *
from torch.utils.data import random_split
[a, _] = random_split(range(44), [44, 0], generator=torch.Generator().manual_seed(42))
idx_pred = a.indices

eeg = []
for subj in range(21):
    tmp = []
    dataSub = loadmat(f"datasetCND_musicImagery/musicImagery/dataCND/dataSub{subj+1}.mat")
    for n in range(88):
        tmp.append(dataSub['eeg']['data'][0,0][0,n].T)
    eeg.append(np.asarray(tmp))
    
eeg = np.asarray(eeg)
for imet in idx_met:
    eeg[:,:,:,imet:imet+34] = -999
a_ma = np.ma.masked_equal(eeg,-999)
eeg = np.array([i.compressed() for i in a_ma]).reshape(21,88,64,1803 - 34*len(idx_met))
    
    
stim = []
tmp = loadmat(f"datasetCND_musicImagery/musicImagery/dataCND/dataStim.mat")
for n in range(88):
    stim.append(tmp['stim']['data'][0,0][1,n])
stim = np.mean(np.asarray(stim), 2)
for imet in idx_met:
    stim[:,imet:imet+34] = -999
a_ma = np.ma.masked_equal(stim,-999)
stim = np.array([i.compressed() for i in a_ma])
# stim[stim!=0] = 1


# stim2 = stim.copy()
# stim2[stim!=0] = 1


time = UTS(tmin=0, tstep=1/64, nsamples=1394)
time.name = 'time'
stim_nd = NDVar(stim[:44,:1394], dims = (Case(44), time))
stim2_nd = NDVar(stim2[:44,:1394], dims = (Case(44), time))


Correct = np.zeros((21,44))
acc = np.zeros((21,))
h_pred, h_pred_null = [],[]
h_img, h_img_null = [],[]
h_lis, h_lis_null = [],[]
for sbj in range(21):
    cor = np.zeros((64,4))
    idx = np.random.choice(44,size=(44,), replace=False)
    print(f'\n Subject: {sbj+1}')
    data_pred =  load.unpickle(f'../Pred_img/sbj{sbj}_Pred.pkl')
    data_img = eeg[sbj,-44:,:,:1394]
    data_lis = eeg[sbj,:44,:,:1394]
    pred_nd = NDVar(data_pred, dims = (Case(44),Sensor.from_montage('biosemi64') ,time))
    img_nd = NDVar(data_img, dims = (Case(44),Sensor.from_montage('biosemi64') ,time))
    lis_nd = NDVar(data_lis, dims = (Case(44),Sensor.from_montage('biosemi64') ,time))
    
    print('\n \t TRF for Predicted Imagined')
    res = boosting(pred_nd, stim_nd[idx_pred], partitions=4,tstart=-.3,tstop=.9)
    h_pred.append(res)
    # res = boosting(pred_nd, stim_nd[idx], partitions=4,tstart=-.3,tstop=.9)
    # h_pred_null.append(res)
    print('\n \t TRF for Imagined')
    res = boosting(img_nd, stim_nd, partitions=4,tstart=-.3,tstop=.9)
    h_img.append(res)
    # res = boosting(img_nd, stim_nd[idx], partitions=4,tstart=-.3,tstop=.9)
    # h_img_null.append(res)
    print('\n \t TRF for Listened')
    res = boosting(lis_nd, stim_nd, partitions=4,tstart=-.3,tstop=.9)
    h_lis.append(res)
    # res = boosting(lis_nd, stim_nd[idx], partitions=4,tstart=-.3,tstop=.9)
    # h_lis_null.append(res)
    y_p2i = convolve(h_pred[sbj].h_scaled,stim_nd[[0,12,23,34]])
    Y = y_p2i.x
    X = img_nd.x
    for itrials in range(44):
        for muse in range(4):
            for ch in range(64):
                cor[ch,muse] = np.corrcoef(Y[muse,ch,:], X[itrials,ch,:])[1,0]
        vote = np.asarray([sum(cor.argmax(1) == i) for i in range(4)]).argmax(0)
        Correct[sbj,itrials] = vote == itrials//11
    acc[sbj] = sum(Correct[sbj])/44
    
print(acc.mean())
#%%    
r_pred = concatenate([res0.r for res0  in h_pred], 'case')
r_pred.name = 'Predicted Imagined'
r_img = concatenate([res0.r for res0  in h_img], 'case')
r_img.name = 'Imagined'
r_lis = concatenate([res0.r for res0 in h_lis], 'case')
r_lis.name = 'Listened'

plot.Topomap([r_lis,r_img,r_pred], clip='circle', vmax = .12)
plot.Topomap([r_img - .06 ,r_pred - .06], clip='circle')

#%%


r_pred = concatenate([res0.h_scaled for res0 in h_pred], 'case')
r_pred.name = 'Predicted Imagined'
r_img = concatenate([res0.h_scaled for res0 in h_img], 'case')
r_img.name = 'Imagined'
#%%
acc_2 = np.zeros((21,))
Correct = np.zeros((21,44))

for sbj in range(21):
    print(f'Subject {sbj+1}')
    cor = np.zeros((64,4))
    data_img = eeg[sbj,-44:,:,:1394]
    data_lis = eeg[sbj,:44,:,:1394]
    img_nd = NDVar(data_img, dims = (Case(44),Sensor.from_montage('biosemi64') ,time))
    lis_nd = NDVar(data_lis, dims = (Case(44),Sensor.from_montage('biosemi64') ,time))
    y_p2i = convolve(h_lis[sbj].h_scaled,stim_nd[[0,12,23,34]])
    Y = y_p2i.x
    X = img_nd.x
    for itrials in range(44):
        for muse in range(4):
            for ch in range(64):
                cor[ch,muse] = np.corrcoef(Y[muse,ch,:], X[itrials,ch,:])[1,0]
        vote = np.asarray([sum(cor.argmax(1) == i) for i in range(4)]).argmax(0)
        Correct[sbj,itrials] = vote == itrials//11
    acc_2[sbj] = sum(Correct[sbj])/44

print(acc_2.mean())
#%%


# Put into dataframe
df = pd.DataFrame({'pred2img': acc, 'lis2img': acc_2})
data = pd.melt(df)

# Plot
fig, ax = plt.subplots()
sb.swarmplot(data=data, x="variable", y="value", ax=ax, s=4, linewidth=1)
sb.violinplot(data=data, x="variable", y="value", ax=ax, linewidth=1)
# violinplot(data=corr_df, palette="Set3", bw=.2, cut=1, linewidth=1)
# Now connect the dots
# Find idx0 and idx1 by inspecting the elements return from ax.get_children()
# ... or find a way to automate it
idx0 = 0
idx1 = 1
locs1 = ax.get_children()[idx0].get_offsets()
locs2 = ax.get_children()[idx1].get_offsets()

# before plotting, we need to sort so that the data points
# correspond to each other as they did in "set1" and "set2"
sort_idxs1 = np.argsort(acc)
sort_idxs2 = np.argsort(acc_2)

# revert "ascending sort" through sort_idxs2.argsort(),
# and then sort into order corresponding with set1
locs2_sorted = locs2[sort_idxs2.argsort()][sort_idxs1]

for i in range(locs1.shape[0]):
    x = [locs1[i, 0], locs2_sorted[i, 0]]
    y = [locs1[i, 1], locs2_sorted[i, 1]]
    ax.plot(x, y, color="black", alpha=0.3)
    
ax.axhline(.25, color='k', linestyle='--', label='chance')