
import torch
from torch.utils.data import Dataset

import numpy as np
import scipy
from scipy import io
import os
import copy


class EncReadDATA(Dataset):

    def __init__(self, execdata, iC, mode, datadir='.', kinfeat=range(13),
                 obsdata=None, iG=2, tasksubj=0, augment=0, transform=None, transform_on=True):
        """
        Args:
            execdata : DataFrame of kinematic data frome execution experiment
            iC : Condition index (1:NDs, 2:PLDs, 0:both)
            mode : 'enc' for intention encoding, 'read' for intention readout
            datadir : the directory containing the data
            kinfeat : indices of kinematic features to extract
            obsdata : DataFrame of observation data (only needed when mode=='read')
            iG : Group index (1:Experts, 2:Controls)
            tasksubj : index of observer within selected group
            augment : number of data replications for augmentation
            transform : optional transform to be applied on a sample.
            transform_on : allows to disable the transform 
                           also when transform is not None.
        """
        
        # Settings
        self.execdata = execdata
        self.iC = iC
        self.mode = mode
        self.datadir = datadir
        self.kinfeat = kinfeat
        self.obsdata = obsdata
        self.iG = iG
        self.tasksubj = tasksubj
        self.augment = augment
        self.transform = transform
        self.transform_on = transform_on
        # Names
        self.condnames = ['CONTROL','EXPERT']
        self.intnames = ['Left', 'Right']
        



        # Load kinematic data
        data = copy.deepcopy(execdata)
        
        # Extract indices for the selected mvt condition from the videos ordered "by encoding"
        # if mode=='read' or returnDuration:
        videonames = data['VIDEO_NAME'].to_numpy()
        # Get the data location by "CONDITION". Here two kinds of video shared the same kinematics so all the condition are 2.
        if iC==1:
            condind = [ii for ii in range(len(videonames)) if data['CONDITION'][ii] == 2] 
        elif iC==2:
            condind = [ii for ii in range(len(videonames)) if data["CONDITION"][ii] == 2] 
        else:
            condind = np.arange(len(videonames)) # all videos
        cvideonames = list(videonames[condind])

        data = data.iloc[condind,:] # select only the videos of the selected condition

        # Extract kinematic variables and intention labels from dataframe
        kindata = data.iloc[:,3:].to_numpy() # from the third cols to the end
        kindata = kindata.reshape((kindata.shape[0],-1,5)) 
        kindata = kindata[:, kinfeat, :] # select only the wanted kinematic features
        intentions = data['OUTCOME'].to_numpy() # intention labels
        subjtype = data['CONDITION'].to_numpy() # subject type (expert or control)



        if mode=='read':
            taskdata = copy.deepcopy(obsdata)
            # Extract wanted observer group and movement group
            taskdata = taskdata.loc[taskdata.loc[:,'SUBJECT_GROUP']==iG,:] # observer group
            if iC!=0:
                taskdata = taskdata.loc[taskdata.loc[:,'CONDITION']==iC,:] # movement group
            # Extract subject number and select wanted subject
            subjnumbers = taskdata['SUBJECT_ID'].unique() # subject IDs for the group
            self.subjn = subjnumbers[tasksubj] # keep track of subject ID
            subjind = taskdata.loc[:,'SUBJECT_ID']==self.subjn
            taskdata = taskdata.loc[subjind,:]
            # Extract subject answers
            stim = taskdata['OUTCOME']
            answ = taskdata['SUBJECT_RESPONSE']

            # Match with kinematic data
            taskvideos = taskdata['VIDEO_NAME'] # ordering of videos for the current task subject
            videos, task_ind, kin_ind = np.intersect1d(taskvideos,
                            cvideonames, return_indices=True) # map subject video order to original order
            answ = np.array(answ)[task_ind] # match with original order
            kindata = kindata[kin_ind,:,:] # possibly remove invalid trials for current subject from kindata
            self.stim = np.array(stim)[task_ind] # match with original order
            self.kin_ind = kin_ind # keep track of invalid trials


        # Final data
        self.kindata = kindata
        if mode=='enc':
            self.target = intentions - 1
        elif mode=='read':
            self.target = answ - 1


        if augment:
            self.kindata = np.tile(self.kindata, (augment,1,1)) # replicate data for augmentation (along the first dimension)
            self.target = np.tile(self.target, augment)


    def __len__(self): # 返回数据集的长度 
        return len(self.target) 


    def __getitem__(self, idx): # 返回数据集中第idx个数据
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = torch.tensor(self.kindata[idx,:,:]) # kinematic data
        
        target = self.target[idx].squeeze()
        

        if self.transform and self.transform_on: # transform the data
            sample = self.transform(sample) # transform the data to tensor

        outputs = sample, target

        return outputs
