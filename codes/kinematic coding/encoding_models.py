
from readout import readout
from utils import driftmodel, compute_pvalues # from utils.py
from encoding import encoding
import pandas as pd

execdata = pd.read_excel('model input.xlsx', sheet_name='Execution')
kinfeat = range(22)     


FeatNames = ["Nose_x", "Nose_y", "Neck_x", "Neck_y", "RShoulder_x", "RShoulder_y", "RElbow_x", "RElbow_y", "RWrist_x", "RWrist_y", "LShoulder_x", "LShoulder_y", "LElbow_x", "LElbow_y", "LWrist_x", "LWrist_y", "MidHip_x", "MidHip_y", "RHip_x", "RHip_y", "LHip_x", "LHip_y"]

def run_encoding(model_params):
    iC, nresample, permtest, cv, verbose, plots = model_params
    
    result = encoding(driftmodel, execdata, kinfeat=kinfeat, iC=iC, nresample=nresample, permtest=permtest, cv=cv, verbose=verbose, plots=plots, FeatNames=FeatNames)

    return result
