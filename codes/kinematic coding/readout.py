
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import copy
import itertools
import numpy as np
import math
import pandas as pd

import os
import sys
sys.path.append('../')

from load_data import EncReadDATA
from utils import TimeWarping


#######################################################################################################


def readout(model, execdata, obsdata, iC, iG, kinfeat=range(13), deltarange=1.5, n_augment=3,
             cv=False, kcv=5, ncv=50, permtest=False, n_perms=200, nresample=1, verbose=1, nsub=1):

    """
    Training and evaluation of the intention readout model.

    INPUTS

        model -- model architecture
        execdata -- DataFrame of kinematic data from execution experiment
        obsdata -- DataFrame of behavioral data from observation experiment
        iC -- movement group (1:NDs, 2:PLDs, 0:both)
        iG -- observer group (1:Expert, 2:Novice)
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
        verbose -- amount of info to print (0: no log; 1: synthetic log; 2: detailed log)


    OUTPUT

        dict, with keys:

        'betas' -- model coefficient vectors
                   size: [Nobservers x Nfeatures], or [Nobservers x Nresample x Nfeatures]
        'train_acc' -- accuracy of the model on the training set
                       size: [Nobservers x 1], or [Nobservers x Nresample]
            (only returned if not cv and not permtest)

        'cvfolds' -- list: for each observer, indices of trials in each cv fold
        'trial_acc' -- list: for each observer, out-of-sample accuracy of the model on each trial
            (only returned if cv)

        'permbetas'-- model coefficients on permuted data
                      size: Nobservers x Nperms+1 x Nfeatures
                  permbetas[subject][0] = coefficients on original data
                  permbetas[subject][1:] = coefficients on permuted data
        'permacc' -- accuracies (cross-validated if cv, on training set otherwise) on permuted data
                     size: Nobservers x Nperms+1
                 permacc[subject][0] = accuracy on original data
                 permacc[subject][1:] = accuracies on permuted data
            (only returned if permtest)

    """

    # SETTINGS
    batch_size = 16
    nepochs = 500
    lambda2 = .05
    epoch_print = 100
    dropout = .25

    #######################################################################################################
    # Setup
    GroupNames = ['BOTH', 'EXPERT','CONTROL'] # observer group names
    CondNames = ['BOTH','Normal Videos','PLDs'] # movement group names

    if verbose: # print settings
        print('\nREADOUT MODEL ON %s KINEMATICS, %s GROUP\n'
              %(CondNames[iC], GroupNames[iG]))


    nsub = nsub # number of observers in each group
    augment = 0 if deltarange==1 else n_augment
    nperms = n_perms if permtest else 0

    if cv:
        nresample = 1

    transform = TimeWarping(deltarange) # time warping transformation

    # initialize lists
    subjID = [] # list of observer IDs
    readbetas = [] # list of model coefficients
    perfmean, perfstd = [], [] # list of mean and std of model performance
    trainperf = []
    cvbetas = [[] for _ in range(nsub)] # list of model coefficients for each cv fold
    cvfolds0 = [[] for _ in range(nsub)]
    trial_acc = [[] for _ in range(nsub)]

    if nresample > 1 :
        betas_resampled = [[] for _ in range(nsub)]
        trainperf_resampled = [[] for _ in range(nsub)]

    if permtest:
        permacc = [[] for _ in range(nsub)]
        permbetas = [[] for _ in range(nsub)]

    # load data
    if verbose>1:
        print('\nLoading %swatch%s data...' %(GroupNames[iG], CondNames[iC]))
    alldata, allwdata = [], []
    if nresample>1:
        tkin, tlab, wkin, wlab = [], [], [], []
    for ts in range(nsub):
        data_tmp = EncReadDATA(execdata, iC=iC, mode='read',
                         kinfeat=kinfeat, obsdata=obsdata,
                         iG=iG, tasksubj=ts)
        alldata.append(data_tmp) # load data. append是把data_tmp加到alldata的最后
        if nresample>1:
            tkin.append(copy.deepcopy(alldata[ts].kindata))
            tlab.append(copy.deepcopy(alldata[ts].target))
        if augment:
            wdata_tmp = EncReadDATA(execdata, iC=iC, mode='read',
                          kinfeat=kinfeat, obsdata=obsdata,
                          iG=iG, tasksubj=ts, transform=transform,
                          augment=augment)
            allwdata.append(wdata_tmp)
            if nresample>1:
                wkin.append(copy.deepcopy(allwdata[ts].kindata))
                wlab.append(copy.deepcopy(allwdata[ts].target))
    if verbose>1:
        print('Done!\n')


    if cv: # set up cross-validation
        foldsize = math.ceil(max([len(alldata[ts]) for ts in range(nsub)])/kcv) # 这行代码的意思是：把每个人的数据分成kcv份，每份的长度是foldsize
        fractions = [i/foldsize for i in range(foldsize+1)] # fractions是一个list，里面的元素是0到foldsize之间的数，每个数之间的间隔是1/foldsize

    timebins = alldata[0].kindata.shape[-1] # number of time bins


    permrange = range(nresample) if nresample > 1 else range(nperms+1)


    # loop over resamples and permutations
    for iperm in permrange:

        if verbose and permtest:
            print('\n========= Permutation #%d/%d %s========='
                      %(iperm,nperms,'(non-permuted data) ' if iperm==0 else ''))
        if verbose and (nresample > 1) :
            print('\n========= Resampling #%d/%d =========' %(iperm+1,nresample))
        if verbose==1:
                print('%s watch %s -- subject # (out of %d): '
                      %(GroupNames[iG], CondNames[iC], nsub)) #, end=' ')

        if permtest and ncv > 5 and iperm > 0:
            ncv = 5


        for tasksubj in range(nsub):

            # training data
            trainset = alldata[tasksubj] # training data is the data from the current observer

            # resample training data
            if nresample > 1 :
                resample = np.random.choice(np.arange(len(trainset)), size=len(trainset))
                wresample = list(copy.deepcopy(resample))
                for i in range(1,augment):
                    wresample = wresample + list(resample+i*len(trainset))
                trainset.kindata = tkin[tasksubj][resample,:,:]
                trainset.target = tlab[tasksubj][resample]

            # permute labels
            if permtest and iperm > 0:
                permidx = np.random.permutation(np.arange(len(trainset))) # permute labels
                trainset.target = trainset.target[permidx]
            temp_loader =  DataLoader(trainset,
                                 shuffle=False, batch_size=len(trainset))
            temp_loader = iter(temp_loader)
            _, lab = temp_loader.__next__()
            ind0 = torch.where(lab==0)[0].numpy()
            ind1 = torch.where(lab==1)[0].numpy()

            # cross-validation
            ratio = len(ind1)/len(trainset) # fraction of lab is 1 in training data

            if cv:
                fold1 = (np.abs(np.array(fractions) - ratio)).argmin() # number of samples with label 1 in each fold
                kcv = min(math.floor(len(ind0)/(foldsize-fold1)), math.floor(len(ind1)/fold1)) # number of folds is the minimum of the number of folds that can be made with label 0 and the number of folds that can be made with label 1
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
                cv_chance = max(fr,1-fr)

                if iperm==0:
                    cvfolds0[tasksubj] = cvfolds

            if augment!=0:
                warpedset = allwdata[tasksubj]
                if nresample > 1 :
                    warpedset.kindata = wkin[tasksubj][wresample,:,:]
                    warpedset.target = wlab[tasksubj][wresample]
                if permtest and iperm > 0:
                    permwidx = []
                    for n in range(augment):
                        permwidx += [ii+n*len(trainset) for ii in permidx]
                    warpedset.target = warpedset.target[permwidx]
                trainset1 = torch.utils.data.ConcatDataset([trainset,warpedset])
            else:
                trainset1 = trainset

            #_?_######### Task subject ID
            if iperm==0:
                subjID.append(trainset.subjn)
            #################

            if verbose>1:
                print(('\n' if not permtest else '\n(perm#%d)   ' %iperm)
                    + '%s kinematics, %s subject %d/%d (%s)\n'
                  %(CondNames[iC],GroupNames[iG],tasksubj+1,nsub,trainset.subjn))
            elif verbose==1:
                print(tasksubj+1, end=' ')


            #_?_###########
            cvacc = []
            cvprobs = {}
            accuracy = {}
            ###############

            for trial in range(ncv*kcv if cv else 1):

                train_idx = np.arange(len(trainset)*(augment+1))

                if cv:
                    cv_idx = cvfolds[trial]
                    to_remove = cv_idx.copy()
                    for idx in range(augment):
                        for icv in cv_idx:
                            to_remove.append((idx+1)*len(trainset)+icv)
                    train_idx = np.delete(train_idx, to_remove)
                    cv_sampler = SubsetRandomSampler(cv_idx)
                    cvloader = DataLoader(trainset, shuffle=False,
                                    sampler=cv_sampler, batch_size=len(cv_idx))


                train_sampler = SubsetRandomSampler(train_idx)
                trainloader = DataLoader(trainset1,
                      sampler=train_sampler, batch_size=batch_size)
                temploader = iter(DataLoader(trainset1,
                      sampler=train_sampler, batch_size=len(train_idx)))
                kindata, _ = temploader.__next__()
                dmean = kindata.mean(0, keepdims=True)
                dstd = kindata.std(0, keepdims=True)


                if verbose>1 and cv:
                    print('\n--------- FOLD #%d%s ---------' %(trial+1,
                        ('' if not permtest else ' (perm#%d)' %iperm)))

                # Initialize the model
                net = model(input_size=len(kinfeat), pdrop=dropout)


                # Train the model

                criterion = nn.BCELoss()

                best_acc = 0.0
                best_epoch = 0

                lossdata = []

                for epoch in range(nepochs):

                    if verbose>1 and (epoch+1) % epoch_print == 0:
                        print('\nFold #%d, Epoch %d' %(trial+1,epoch+1) if cv
                              else 'Epoch %d' %(epoch+1))


                    if epoch - best_epoch > 100:
                        # if training accuracy does not increase for 100 epochs, break
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


                    lossdata.append(epoch_loss/(i+1)) # divided by the number of batches (final value of i+1)
                    epoch_acc = epoch_corrects / len(train_idx) # divided by the number of examples
                    if verbose>1 and (epoch+1) % epoch_print == 0:
                        print('train accuracy: %.3f\n' %epoch_acc)
                    # update best epoch
                    if epoch_acc > best_acc:
                        best_epoch = epoch
                        best_acc = epoch_acc


                if verbose>1:
                    print('Finished training' + (' for fold #%d' %(trial+1) if cv else '')
                        +(': best train accuracy = %.3f at epoch %d\n' %(best_acc, best_epoch+1))
                     )
                trainperf.append(epoch_acc)
                if nresample>1:
                    trainperf_resampled[tasksubj].append(epoch_acc)


                # model weights
                betas = net.fc.weight.detach().numpy().squeeze()
                if nresample > 1 :
                    betas_resampled[tasksubj].append(betas)
                if cv and iperm==0:
                    cvbetas[tasksubj].append(betas)


                if cv:
                    net.eval()  # set to evaluation mode
                    trainset.transform_on = False # don't apply noise transform

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
                        preds = (outputs.squeeze() > .5)
                        total += labels.shape[0]
                        correct += np.sum((preds==labels).numpy())
                        trial_acc[tasksubj].append((preds==labels).numpy())
                        cvacc.append(correct/total)
                        if verbose>1:
                            print('Evaluation on out-of-sample elements: %.3f' %(correct/total))
                            print('Mean perf until now = %.3f\n' %np.mean(cvacc))

                    trainset.transform_on = True # restore noise transform


            if permtest and cv:
                permacc[tasksubj].append(np.mean(cvacc))
            elif permtest and not cv:
                permacc[tasksubj].append(epoch_acc)
                permbetas[tasksubj].append(betas)
            elif not permtest and not cv:
                readbetas.append(betas)


            if cv:
                if verbose>1:
                    print('\nReadout model on %s kinematics, %s subject %d/%d (%s)\ncv performance: %.3f+-%.3f\n\n---\n'
                        %(GroupNames[iC],GroupNames[iG],tasksubj+1,nsub,trainset.subjn,
                          np.mean(cvacc),np.std(cvacc)/math.sqrt(len(cvacc))))
                if iperm==0:
                    perfmean.append(np.mean(cvacc))
                    perfstd.append(np.std(cvacc))


            if cv and (iperm==0):
                trial_acc0 = trial_acc.copy()

    #######################################################################################################
    # Save data and print info


    if verbose==1:
        print()
    if verbose:
        print('\nReadout model on %s kinematics, %s group \n' %(CondNames[iC],GroupNames[iG]))

        for tasksubj in range(nsub):
            if cv:
                print('%s task subject %s : mean accuracy over CV folds = %.1f%%'
                 %(GroupNames[iG],subjID[tasksubj],perfmean[tasksubj]*100))
            else:
                print('%s subject %s : accuracy on training = %.1f%%'
                 %(GroupNames[iG],subjID[tasksubj],trainperf[tasksubj]*100))


    outputs = {}
    if not cv and not permtest:
        outputs['betas'] = np.array(betas_resampled) if nresample>1 else np.array(readbetas)
        outputs['train_acc'] = np.array(trainperf_resampled) if nresample>1 else np.array(trainperf)
    elif cv:
        outputs['cvfolds'] = cvfolds0
        outputs['trial_acc'] = trial_acc0
    elif permtest:
        outputs['permbetas'] = np.array(permbetas)
        outputs['permacc'] = np.array(permacc)

    return outputs
