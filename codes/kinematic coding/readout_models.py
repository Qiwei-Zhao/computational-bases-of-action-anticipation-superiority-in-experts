
from readout import readout
from utils import driftmodel, compute_pvalues # from utils.py
from encoding import encoding
import pandas as pd

# load data

execdata = pd.read_excel('model input.xlsx', sheet_name='Execution')
obsdata = pd.read_excel('model input.xlsx', sheet_name='Observation')
kinfeat = range(22)     


def run_readout(model_params):
    iC, iG, nsub, nresample, permtest, cv, verbose = model_params
    
    result = readout(driftmodel, execdata, obsdata, kinfeat=kinfeat, iC=iC, iG=iG, nsub=nsub, nresample=nresample, permtest=permtest, cv=cv, verbose=verbose)

    return result
