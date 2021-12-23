#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 15:42:56 2021

@author: mohsenr
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from eelbrain import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator, LinearModel, SlidingEstimator
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
#%%
r_model = []
r_null = []
data = []
scores = np.zeros((21,1394,1394))
for subj in range(21):
    tmp = []
    dataSub = loadmat(f"datasetCND_musicImagery/musicImagery/dataCND/dataSub{subj+1}.mat")
    for n in range(88):
        tmp.append(dataSub['eeg']['data'][0,0][0,n].T)
    data.append(np.asarray(tmp))
    
data = np.asarray(data)


chorals = np.asarray([i//11 for i in range(44)])
confusion = np.zeros((21, len(set(chorals)), len(set(chorals))))
met = loadmat('datasetCND_musicImagery/musicImagery/dataCND/metronome.mat')
met = met['met']
idx_met = np.where(met[:,0] != 0)[0]

for imet in idx_met:
    data[:,:,:,imet:imet+34] = 0

a_ma = np.ma.masked_equal(data,0)
data = np.array([i.compressed() for i in a_ma]).reshape(21,88,64,1803 - 34*len(idx_met))

#%%

device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class Lis2Img(nn.Module):
    def __init__(self):
        super().__init__()
        # self.subject_embedding = nn.Embedding(21, 16)
        self.encoder = nn.Sequential(
            ########nn.Conv1d(64+16, 64, 4, stride=2, padding=2),
            nn.Conv1d(64, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32,32, 4, stride=2, padding=2),
            nn.ReLU(),

        )
        # self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 4, stride=2, padding=2)
        )

    def forward(self, x):
        # x has shape [4, 64, 1803]
        # subj_id has shape [4]

        # subj = self.subject_embedding(subj_id)              # subj has shape [4, 16]
        # # Copy embedding vector into all time points
        # subj = subj.view(4, -1, 1).expand(-1, -1, 1802)     # subj has shape [4, 16, 1803]
        # Now, subj[:, :, 0] == subj[:, :, 1] == .... == subj[:, :, 1801]

        # Append the 16 subject embedding channels onto the 64 EEG channels
        ########x = torch.cat((x, subj), dim=1)
        # x now has shape [4, 64+16, 1803]


        # Conv1d input shape: [batch size, # channels, signal length]
        x = self.encoder(x)

        # LSTM input shape: [batch size, signal length, # features] -- when batch_first=True!
        x = x.transpose(1, 2)
        # x, _ = self.lstm(x)
        x = x.transpose(2, 1)

        x = self.decoder(x)
        return x

# model = Lis2Img().to(device)
# model.to(dtype=torch.double)
# print(model)

# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def R_value(a, b):                      # Inputs shape [4, 64, 1803]
    dim = -1                            # Correlation over time points dimension
    a = a - a.mean(axis=dim, keepdim=True)
    b = b - b.mean(axis=dim, keepdim=True)
    cov = (a * b).mean(dim)
    na, nb = [(i**2).mean(dim)**0.5 for i in [a, b]]
    norms = na * nb
    R_matrix = cov / norms
    return R_matrix                     # [4, 64] - R values per channel for each trial


def train(x_dataloader, y_dataloader, model, loss_fn, optimizer):
    assert (len(x_dataloader.dataset) == len(y_dataloader.dataset)), \
                "Input, output dataloaders have different lengths! :O"

    size = len(x_dataloader.dataset)
    for batch, (X, Y) in enumerate(zip(x_dataloader, y_dataloader)):
        X, Y = X.to(device), Y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"[{current:>4d}/{size:>4d}]   loss: {loss}")

def test(x_dataloader, y_dataloader, model, loss_fn):
    assert (len(x_dataloader) == len(y_dataloader)), \
                "Input, output dataloaders have different lengths! :O"

    num_batches = len(x_dataloader)
    model.eval()
    avg_loss = 0
    R_values = []

    with torch.no_grad():
        for (X, Y) in zip(x_dataloader, y_dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, Y).item()
            R = R_value(pred, Y)
            R_values.extend(R.flatten().tolist())

    avg_loss /= num_batches
    avg_R = np.asarray(R_values).mean()
    print("Test:")
    print(f"\tAvg loss: {avg_loss}")
    print(f"\tAvg R value: {avg_R}\n")
    

acc = np.zeros((21,))
for test_sbj_idx in range(0,21):
    # model.apply(reset_weights)
    
    # loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ##
    model = Lis2Img().to(device)
    model.to(dtype=torch.double)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ##
    
    train_sbjs_idx = list(range(21))
    train_sbjs_idx.remove(test_sbj_idx)
    
    train_sbjs = data[train_sbjs_idx,:,:,:1394]#:1802]
    test_sbj = data[test_sbj_idx,:,:,:1394]#:1802]
    
    # (1) make data to pre-train the model as and AE for 20 subjects (all data from 20 sbjs)
    # (2) fix the encoder train the decoder part for lis_to_img (devide into train and test)
    # (3) test the network for the left-out sbj (all data from left_out sbjs)
    
    data_tr = train_sbjs.reshape(20*88,64,1394)                     #(1)
    data_lis = train_sbjs[:,:44,:,:].reshape(20*44,64,1394)         #(2a)
    data_img = train_sbjs[:,-44:,:,:].reshape(20*44,64,1394)        #(2b)
    tst_sbj_lis = test_sbj[:44,:,:]                                 #(3a)
    tst_sbj_img = test_sbj[-44:,:,:]                                #(3b)
    
    [all_train, all_test] = random_split(data_tr, [1700, 60],        #(1)
                                         generator=torch.Generator().manual_seed(42)) 
    
    [lis_train, lis_test] = random_split(data_lis, [800, 80],       #(2)
                                 generator=torch.Generator().manual_seed(42))
    [img_train, img_test] = random_split(data_img, [800, 80],        #(2))
                                 generator=torch.Generator().manual_seed(42))
    
    
    # Create dataloaders
    batch_size = 4
    all_train_dataloader = DataLoader(all_train, batch_size=batch_size)
    all_test_dataloader = DataLoader(all_test, batch_size=batch_size)
    
    lis_train_dataloader = DataLoader(lis_train, batch_size=batch_size)
    lis_test_dataloader = DataLoader(lis_test, batch_size=batch_size)
    img_train_dataloader = DataLoader(img_train, batch_size=batch_size)
    img_test_dataloader = DataLoader(img_test, batch_size=batch_size)
    
    

    epochs = 30
    print("[[Pre-training autoencoder]]")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(all_train_dataloader, all_train_dataloader, model, loss_fn, optimizer)
        test(all_test_dataloader, all_test_dataloader, model, loss_fn)
        
    
    print("[[Fix decoder weights]]")
    model.encoder.requires_grad_(False)  
    epochs = 30
    print("[[Pre-training autoencoder]]")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(lis_train_dataloader, img_train_dataloader, model, loss_fn, optimizer)
        test(lis_test_dataloader, img_test_dataloader, model, loss_fn)
    print("Done!\n")
    
    
    ###### lis2img for the unseen subject
    [test_sbj_lis, _] = random_split(tst_sbj_lis, [44, 0], generator=torch.Generator().manual_seed(42))
    [test_sbj_img, _] = random_split(tst_sbj_img, [44, 0], generator=torch.Generator().manual_seed(42))
    test_lis = DataLoader(test_sbj_lis, batch_size=batch_size)
    test_img = DataLoader(test_sbj_img, batch_size=batch_size)
    print(f'Test the network on Subject: {test_sbj_idx + 1}')
    test(test_lis, test_img, model, loss_fn)    
    
    X = torch.tensor(test_sbj_lis)
    Y = torch.tensor(test_sbj_img)
    Pred = model(X).detach()
    # tmp = R_value(Y,Pred)
    # r_model.append(np.asarray(tmp))
    # save.pickle(np.asarray(Pred), f'../Pred_img/sbj{test_sbj_idx}_Pred.pkl')
    ############ Linear Classification    
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                    LinearModel(LogisticRegression(penalty = 'l2', C=1, 
                                solver = 'liblinear', class_weight='balanced')))
    decoder = SlidingEstimator(clf,scoring=None,n_jobs=-1,verbose=True)
    X = np.asarray(Pred)
    y = chorals[test_sbj_img.indices]
    classes = set(y)
    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    ## Compute confusion matrix for each cross-validation fold
    y_pred = np.zeros((len(y), 1394 ,len(classes)))
    # decoder.fit(X = np.asarray(Y), y = chorals[test_sbj_img.indices])
    # scores[test_sbj_idx,:,:] = decoder.score(X = np.asarray(Pred), y = chorals[test_sbj_img.indices])
    for train_t, test_t in cv.split(X, y):
        # Fit
        decoder.fit(X[train_t], y[train_t])
        # Probabilistic prediction (necessary for ROC-AUC scoring metric)
        y_pred[test_t] = decoder.predict_proba(X[test_t])
        
        
    tmp = y_pred.mean(axis = 1)
    acc[test_sbj_idx] = sum(tmp.argmax(1) == y)/44
    for ii, train_class in enumerate(classes):
        for jj in range(ii, len(classes)):
            confusion[test_sbj_idx, ii, jj] = roc_auc_score(y == train_class,tmp[:, jj])
            confusion[test_sbj_idx, jj, ii] = confusion[test_sbj_idx, ii, jj]
    
    # fig, ax = plt.subplots(1)
    # im = ax.matshow(scores[test_sbj_idx,:,:], vmin=.0, vmax=.5,
    #                 cmap='RdBu_r', origin='lower',)# extent = [0, 100, 0, 100])
    # # ax.contour(p_val, levels = [-0.1, 0.05], colors = 'k', linestyles = '--', 
    # #              linewidths =.35)
    # ax.axhline(0., color='k')
    # ax.axvline(0., color='k')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.set_xlabel('Testing Time (s)')
    # ax.set_ylabel('Training Time (s)')
    # ax.set_title(f'train on True, test on Pred: sbject{test_sbj_idx+1}')
    # clb = plt.colorbar(im)
    # clb.set_label('Accuracy  (chance = 0.25)')
    # plt.tight_layout()
    
    labels = [f'Choral {i}' for i in range(1,5)]
    fig, ax = plt.subplots(1)
    im = ax.matshow(confusion[test_sbj_idx,:,:], cmap='RdBu_r', clim=[0., 1])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_title(f'predicted Imagined Subject {test_sbj_idx+1}_{round(acc[test_sbj_idx],2)}')
    # ax.axhline(4, color='k')
    # ax.axvline(4, color='k')
    clt = plt.colorbar(im)
    clt.set_label('AUC')
    plt.tight_layout()
    plt.show()
    
    
    ###### lis2img for control
    [test_sbj_lis, _] = random_split(tst_sbj_lis, [44, 0], generator=torch.Generator().manual_seed(4))
    [test_sbj_img, _] = random_split(tst_sbj_img, [44, 0], generator=torch.Generator().manual_seed(42))
    test_lis = DataLoader(test_sbj_lis, batch_size=batch_size)
    test_img = DataLoader(test_sbj_img, batch_size=batch_size)
    print(f'Test the network on CONTROL Subject: {test_sbj_idx + 1}')
    test(test_lis, test_img, model, loss_fn)
        
    X = torch.tensor(test_sbj_lis)
    Y = torch.tensor(test_sbj_img)
    Pred = model(X).detach()
    tmp = R_value(Y,Pred)
    r_null.append(np.asarray(tmp))
    
    
    
    
    
    
#%%

idx = np.argsort(r_model.mean(1))
idx = idx[::-1]
plt.scatter(np.arange(1,23,1),r_model[idx,:].mean(axis = 1))
plt.errorbar(np.arange(1,23,1),r_model[idx,:].mean(axis = 1),
             yerr=r_model[idx,:].std(axis = 1)/np.sqrt(21))
plt.scatter(np.arange(1,23,1),r_null[idx,:].mean(axis = 1))
plt.errorbar(np.arange(1,23,1),r_null[idx,:].mean(axis = 1),
             yerr=r_null[idx,:].std(axis = 1)/np.sqrt(21))
plt.xticks(np.arange(1, 23, step=1))
plt.tight_layout()
# plt.savefig('../Figures/Participants_N-1.pdf')