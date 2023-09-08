
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import copy
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools 
import pandas as pd

import os
import sys
sys.path.append('../')

from load_data import EncReadDATA
from utils import TimeWarping


#######################################################################################################


def encoding(model, execdata, FeatNames, iC, kinfeat=range(13), deltarange=1.5, n_augment=3,
             cv=False, kcv=5, ncv=50, permtest=False, n_perms=200, nresample=1,
             plots=False, verbose=1):
    """
    Training and evaluation of the intention encoding model.

    INPUTS

        model -- model architecture
        execdata -- DataFrame of kinematic data from execution experiment
        iC -- movement group (1:NDs, 2:PLDs, 0:both)
        kinfeat -- indices of kinematic features to include
        deltarange -- if different from 1, data will be augmented via
                    time warping with delta in [1/deltarange, deltarange]
        n_augment -- number of replications of the data for augmentation (if deltarange>1)
        cv -- if True, evaluate model via repeated k-fold cross-validation
        kcv -- number of folds for cross-validation
        ncv -- number of repetitions for cross-validation
        permtest -- if True, train model on permuted data after training on original data
        n_perms -- number of permutations for permutation test
        nresample -- if >1, take several resamples of the data (turns off if cv or permtest)
        plots -- if True, plot evolution of loss function and model coefficients (turns off if permtest or cv)
        verbose -- amount of info to print (0: no log; 1: synthetic log; 2: detailed log)


    OUTPUT

        dict, with keys:

        'betas' -- model coefficient vector, or list of coefficient vectors fitted on resampled data
            (only returned if not cv and not permtest)

        'cvfolds' -- indices of trials in each cv fold
        'trial_acc' -- out-of-sample accuracy of the model on each trial
            (only returned if cv)

        'permbetas'-- model coefficients on permuted data
                   permbetas[0] : coefficients on original data
                   permbetas[1:] : coefficients on permuted data
            (only returned if permtest and not cv)
        'permacc' -- accuracies (cross-validated if cv, on training set otherwise) on permuted data
                  permacc[0] : accuracy on original data
                  permacc[1:] : accuracies on permuted data
            (only returned if permtest)

    """

    # SETTINGS
    batch_size = 16
    nepochs = 500
    lambda2 = .05
    dropout = .25
    epoch_print = 100

    #######################################################################################################
    # Setup

    FeatNames = FeatNames
    CondNames = ['BOTH','NDs','PLDs']
    Nf = len(kinfeat) # length of kinematic features

    if cv or permtest:
        nresample = 1
        plots = False

    nperms = n_perms if permtest else 0

    augment = 0 if deltarange==1 else n_augment
    transform = TimeWarping(deltarange)

    trainset = EncReadDATA(execdata, iC=iC, mode='enc', kinfeat=kinfeat)
    if augment:
        warpedset = EncReadDATA(execdata, iC=iC, mode='enc',kinfeat=kinfeat,
                                augment=augment, transform=transform) # transform

    timebins = trainset.kindata.shape[-1]

    if cv:
        foldsize = math.ceil(len(trainset)/kcv)
        fractions = [i/foldsize for i in range(foldsize+1)]



    ###########################################
    # create copies of trainset to sample from
    if nresample > 1:
        tkin = copy.deepcopy(trainset.kindata) # kinematic data
        tlab = copy.deepcopy(trainset.target) # intentions
        if augment:
            wkin = copy.deepcopy(warpedset.kindata)
            wlab = copy.deepcopy(warpedset.target)
    ###########################################



    if permtest:
        permacc = [] 
        permbetas = np.zeros((nperms+1,len(kinfeat)))

    trial_acc = []

    if nresample > 1:
        betas_resampled = []
        permrange = range(nresample)
    else:
        permrange = range(nperms+1)

    if verbose:
        print('\nENCODING MODEL ON %s DATASET\n' %(CondNames[iC]))
    if verbose==1 and (permtest or (nresample>1)):
        print('%s (out of %d): ' %('Permutation' if permtest else 'Resample',
                                    nperms if permtest else nresample))
    elif verbose==1 and cv:
        print('Fold (out of %d): ' %(kcv*ncv), end=' ')

    for iperm in permrange:

        if permtest and ncv > 5 and iperm > 0:
            ncv = 5

        if nresample > 1:
            if verbose>1:
                print('========= Resampling #%d/%d =========' %(iperm+1,nresample))
            elif verbose==1:
                print(' '*(3-len(str(iperm+1)))+str(iperm+1),
                      end=(' ' if (iperm+1)%20 else '\n'))
            resample = np.random.choice(np.arange(len(trainset)), size=len(trainset))
            trainset.kindata = tkin[resample,:,:]
            trainset.target = tlab[resample]
            if augment:
                wresample = list(copy.deepcopy(resample))
                for i in range(1,augment):
                    wresample = wresample + list(resample+i*len(trainset))
                warpedset.kindata = wkin[wresample,:,:]
                warpedset.target = wlab[wresample]

        if permtest and verbose>1:
            print('========= Permutation #%d/%d %s========='
                  %(iperm,nperms,'(non-permuted data) ' if iperm==0 else ''))
        elif permtest and (verbose==1):
                print(' '*(3-len(str(iperm)))+str(iperm)
                      +(' (non-permuted data)' if iperm==0 else ''),
                      end=(' ' if iperm%20 else '\n'))


        if permtest and (iperm > 0):
            permidx = np.random.permutation(np.arange(len(trainset))) # create the permutation labels
            trainset.target = trainset.target[permidx]
            if augment:
                permdriftidx = np.tile(permidx, augment)
                warpedset.target = warpedset.target[permdriftidx]

        temp_loader =  DataLoader(trainset,
                                 shuffle=False, batch_size=len(trainset))
        temp_loader = iter(temp_loader)
        alltraindata, lab = temp_loader.__next__()
        ind0 = torch.where(lab==0)[0] # place action
        ind1 = torch.where(lab==1)[0] # pour action

        ratio = len(ind1)/len(trainset) # fraction of Pouring responses

        if cv:
            # create class-balanced CV folds
            fold1 = (np.abs(np.array(fractions) - ratio)).argmin() # number of Pouring samples in CV fold
            kcv = min(math.floor(len(ind0)/(foldsize-fold1)), math.floor(len(ind1)/fold1))
            cvfolds = []
            for n in range(ncv): 
                shuffled0 = np.random.permutation(ind0)
                shuffled1 = np.random.permutation(ind1)
                for f in range(kcv):
                    cvfolds.append(list(shuffled0[(foldsize-fold1)*f:(foldsize-fold1)*(f+1)])\
                              + list(shuffled1[fold1*f:fold1*(f+1)]))
                if (foldsize-fold1)*(f+1)<len(shuffled0) or fold1*(f+1)<len(ind1):
                    cvfolds.append(list(shuffled0[(foldsize-fold1)*(f+1):])\
                                  + list(shuffled1[fold1*(f+1):]))
                    kcv += 1 if n==ncv-1 else 0
            fr = fractions[fold1]

            if iperm==0:
                cvfolds0 = cvfolds


        if augment:
            trainset1 = torch.utils.data.ConcatDataset([trainset,warpedset])
        else:
            trainset1 = trainset


        ###################################################################################################
        # TRAIN MODEL

        cvacc = []
        cvbetas = []


        for trial in range(ncv*kcv if cv else 1):

            train_idx = np.arange(len(trainset)*(augment+1))

            if cv:
                cv_idx = cvfolds[trial]
                to_remove = cv_idx.copy()
                for idx in range(augment):
                    for icv in cv_idx:
                        to_remove.append((idx+1)*(len(trainset)-1)+icv)
                train_idx = np.delete(train_idx, to_remove)
                cv_sampler = SubsetRandomSampler(cv_idx)
                cvloader = DataLoader(trainset,
                                    sampler=cv_sampler, batch_size=len(cv_idx))

            train_sampler = SubsetRandomSampler(train_idx)
            trainloader =  DataLoader(trainset1,
                  sampler=train_sampler, batch_size=batch_size)

            temploader = iter(DataLoader(trainset1,
                      sampler=train_sampler, batch_size=len(train_idx)))
            kindata, _ = temploader.__next__()
            dmean = kindata.mean(0, keepdims=True)
            dstd = kindata.std(0, keepdims=True)


            if cv and verbose>1:
                print('\n--------- FOLD #%d%s ---------' %(trial+1,
                    ('' if not permtest else ' (perm#%d)' %iperm)))
            elif cv and verbose==1:
                print(trial+1, end=' ')

            # Initialize the model
            net = model(input_size=len(kinfeat), pdrop=dropout)


            # Train the model

            criterion = nn.BCELoss()

            #_?_#################
            best_acc = 0.0
            best_epoch = 0
            last_epoch = nepochs

            lossdata = []
            #####################


            for epoch in range(nepochs):

                if verbose>1 and ((epoch+1) % epoch_print == 0):
                    print('\nFold #%d, Epoch %d' %(trial+1,epoch+1) if cv
                          else 'Epoch %d' %(epoch+1))


                if epoch - best_epoch > 100:
                    # if training accuracy does not increase for 100 epochs, break
                    last_epoch = epoch
                    break

                optimizer = optim.Adam(net.parameters(), weight_decay=lambda2)

                net.train(True)  # Set model to training mode

                epoch_loss = 0.0
                epoch_corrects = 0

                for i, data in enumerate(trainloader, 0):

                    # get the inputs
                    inputs, labels = data
                    inputs = (inputs - dmean)/dstd
                    inputs, labels  = inputs.float(), labels.float()

                    outputs = torch.tensor([1/2]).float()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    for t in range(timebins):
                        outputs = net(t, inputs[:,:,t], outputs)
                    preds = outputs>.5

                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    loss.backward()
                    optimizer.step()

                    # statistics
                    epoch_loss += loss.item()
                    epoch_corrects += torch.sum(preds.T == labels).item()

                lossdata.append(epoch_loss/(i+1)) # divided by the number of batches (final value of i)
                epoch_acc = epoch_corrects / len(train_idx) # divided by the number of examples
                if verbose>1 and ((epoch+1) % epoch_print == 0):
                    print('train accuracy: %.3f\n' %(epoch_acc))
                # deep copy the model
                if epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc


            if verbose>1:
                print('Finished training' + (' for fold #%d' %(trial+1) if cv else '')
                    +(': best train accuracy = %.3f at epoch %d' %(best_acc, best_epoch+1))
                 )
                print()

            betas = net.fc.weight.detach().numpy().squeeze()
            if permtest and not cv:
                permbetas[iperm,:] = betas
            if nresample > 1:
                betas_resampled.append(betas)
            if cv:
                cvbetas.append(betas)

            if cv:
                net.eval() # set to evaluation mode
                trainset.transform_on = False # don't apply transform

                # Evaluate the net on out-of-sample
                total = 0
                correct = 0
                for data in cvloader:
                    inputs, labels = data
                    inputs = (inputs - dmean)/dstd
                    inputs, labels = inputs.float(), labels.float()
                    outputs = torch.tensor([1/2]).float()
                    for t in range(timebins):
                        outputs = net(t, inputs[:,:,t], outputs)
                    preds = outputs.squeeze() > .5
                    total += labels.shape[0]
                    correct += np.sum((preds==labels).numpy())
                    trial_acc.append((preds==labels).numpy())
                tempacc = correct/total
                cvacc.append(tempacc)
                if verbose>1:
                    print('Evaluation on out-of-sample elements: %.3f' %(correct/total))
                    print('Mean CV perf until now = %.3f\n' %(np.mean(cvacc)))

                trainset.transform_on = True # restore transform


        if permtest and cv:
            permacc.append(np.mean(cvacc))
        elif permtest and not cv:
            #_?_####################
            permacc.append(best_acc)
            ########################

        if cv and iperm==0:
            trial_acc0 = trial_acc.copy()


    #######################################################################################################
    # Final results and plots

    if plots:

        if nresample==1:
            # Loss plot
            vepochs = np.arange(1,last_epoch+1)
            lossplot = plt.figure()
            train_loss, = plt.plot(vepochs, lossdata, label='Train loss')
            plt.legend(handles=[train_loss,])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('%s - Loss plot' %(model.__name__))
            plt.grid(True)


            meanfeat = betas
        else:
            confint = [np.quantile([b[fe] for b in betas_resampled], [.025,.975]) for fe in range(Nf)]
            meanfeat = [np.mean([b[fe] for b in betas_resampled]) for fe in range(Nf)]
            rel_feat = [fe for fe in range(Nf) if confint[fe][0]*confint[fe][1]>0]
            err = np.array([np.array(meanfeat)-np.array(confint)[:,0],
                        np.array(confint)[:,1]-np.array(meanfeat)])

        ind = np.argsort(np.abs(meanfeat))[::-1]
        mf_sorted = [meanfeat[ii] for ii in ind]
        names_sorted = [FeatNames[ii] for ii in ind]

        f = plt.figure()
        ax = f.add_subplot(111)
        plt.bar(np.arange(Nf), mf_sorted, color='lightgray', width=1,
                edgecolor='w', linewidth=2)
        if nresample>1:
            err_sorted = [[err[0][ii] for ii in ind], [err[1][ii] for ii in ind]]
            plt.errorbar(np.arange(Nf), mf_sorted, yerr=err_sorted,
            linewidth=0, elinewidth=1, color='k')
        plt.axhline(0, color='k', linewidth=1)
        plt.xticks(np.arange(Nf), names_sorted, rotation='vertical', fontname='Arial', fontsize=18)

    if verbose==1:
        print()
    if verbose:

        if cv:
            print('\nEncoding model on %s DATASET: mean accuracy over CV folds = %.1f%%'
                  %(CondNames[iC],np.mean(trial_acc0)*100))

        # Warnings
        if deltarange != 1:
            print('\nNOTE : trained with data augmentation')
        else:
            print('\nNOTE : trained with NO data augmentation')
        if nresample>1:
            print('\nNOTE : results over %d resamplings of the data' %nresample)

    outputs = {}
    if not cv and not permtest:
        outputs['betas'] = betas_resampled if nresample>1 else betas
    if cv:
        outputs['cvfolds'] = cvfolds0
        outputs['trial_acc'] = trial_acc0
    if permtest:
        outputs['permbetas'] = permbetas
        outputs['permacc'] = permacc

    return outputs
