
# INOUT: Raw Datas from E-Prime, Raw Kinematic Data from openpose, Data from kinematic coding analysis
# OUTPUT: Raw Behavioral Data, Cleaned Behavioral Data, Model Input Data, Statistics Data

# Steps:
# 1. Raw Kinematics Data Processing
# 2. Behavioral Data Transformation
# 3. Behavioral Data Preprocessing
# 4. Statistics Data Preparation
# 5. Corrs-tasks overlap & alinment



# region 1. Raw Kinematics Data Processing

import pandas as pd

# 读取Excel文件
df = pd.read_excel('data/openpose_output.xlsx')

# 1. 删除所有置信分数列
df = df.loc[:, ~df.columns.str.endswith('_confidence')]

# 2. 仅保留所需列
columns_to_keep = ['No', 'frame', 'Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y', 'RElbow_x', 'RElbow_y', 'RWrist_x', 'RWrist_y', 'LShoulder_x', 'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y', 'MidHip_x', 'MidHip_y', 'RHip_x', 'RHip_y', 'LHip_x', 'LHip_y']
df = df[columns_to_keep]

# 3. orginal data with 25 frames
df_raw = df.copy()
df_raw = df_raw.groupby(['No', 'frame']).mean().reset_index()
df_raw['frame'] = df_raw['frame'].astype(int)
# delete the rows with frame = 26
df_raw = df_raw[df_raw['frame'] != 26]
df_raw.rename(columns={'No': 'VIDEO_NAME'}, inplace=True)
# import the "CONDITION" column
df_condition = pd.read_excel('PreExP_Model_INPUT.xlsx', usecols=['VIDEO_NAME', 'OUTCOME'])
# merge the two dataframes using the VIDEO_NAME
df_raw = pd.merge(df_raw, df_condition, on='VIDEO_NAME', how='left')
# save the raw data
df_raw.to_csv('raw_kinematics.csv', index=False)


# 4. 对于frame列中的数据(1-25)，每5行进行平均，即将25行变成5行，并重新命名frame为1-5
df['frame'] = pd.cut(df['frame'], bins=range(0, 26, 5), labels=['1', '2', '3', '4', '5'])
df = df.groupby(['No', 'frame']).mean().reset_index() # 重置索引
df['frame'] = df['frame'].astype(int)

# 5. 按照frame列，将行转换成列，frame中的数字做为后缀
df = df.pivot(index='No', columns='frame')
df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]

# 6. 保留No列，命名为"VIDEO_NAME"，按照No的顺序排行序
df.reset_index(inplace=True)
df.rename(columns={'No': 'VIDEO_NAME'}, inplace=True)
df = df.sort_values('VIDEO_NAME')


# 6. 读取文件“Model_INPUT.xlsx”中的“CONDITION”和“OUTCOME”列”，将它们插入到原有的df中，列的位置是第2、3列，列名不变
condition_outcome_df = pd.read_excel('Model_INPUT.xlsx', usecols=['CONDITION', 'OUTCOME'])
df = pd.concat([df[['VIDEO_NAME']], condition_outcome_df, df.drop(['VIDEO_NAME'], axis=1)], axis=1)

# 7. 保存df为.xlsx，sheet名为“Execution”，文件名为“Model_INPUT_openpose”
df.to_excel('model input openpose.xlsx', sheet_name='Execution', index=False)


# endregion



# region 2. Behavioral Data Transformation(Raw data files -> One merged raw date)

import subprocess
import os
import pandas as pd


# Check if the convert-eprime package is installed
try:
    import convert_eprime
except ImportError:
    # Install the convert-eprime package using pip
    subprocess.check_call(["pip", "install", "git+https://github.com/tsalo/convert-eprime.git"])
    import convert_eprime

from convert_eprime import text_to_csv


# Define a function to convert E-Prime text files to CSV format
def convert_eprime_txt(prefix, current_dir=os.getcwd(), output_postfix="_BehavioralRaw"):
    """
    This function converts E-Prime text files to CSV format. The function looks for text files in the current working directory that start with the input_file_prefix and have a .txt extension. It then converts each file to a CSV file with the same name as the input file. If the output file already exists, the function skips that file and moves on to the next one.
    
    Inputs:
    prefix (str): The prefix of the input file names.
    current_dir (str): The path to the directory containing the input files. Defaults to the current working directory.
    output_postfix (str): The postfix to add to the output file name. Defaults to "_BehavioralRaw".

    Outputs:
    Raw Behavioral Data in CSV format
    
    """

    # Get the current working directory
    current_dir = current_dir

    # Find all text files in current directory that start with the given prefix and end with .txt
    text_files = [f for f in os.listdir(current_dir) if f.startswith(prefix) and f.endswith(".txt")]

    # Loop through the text files and convert each one to a csv
    for file in text_files:
        # Create the full file path
        file_path = os.path.join(current_dir, file)

        # Create the output file path
        out_file = os.path.splitext(file_path)[0] + ".csv"

        # Check if the output file already exists
        if os.path.exists(out_file):
            print(f"Skipping {out_file} because it already exists")
        else:
            # Convert the file to csv using the text_to_csv function
            text_to_csv(file_path, out_file)

    # Concatenate all converted CSV files into one output file
    csv_files = [f for f in os.listdir(current_dir) if f.startswith(prefix) and f.endswith(".csv")]
    if len(csv_files) > 0:
        combined_csv = pd.concat([pd.read_csv(os.path.join(current_dir, f)) for f in csv_files])
        combined_csv.to_csv(f"{current_dir}/{prefix}{output_postfix}.csv", index=False)
        print(f"Combined CSV file saved as {prefix}_{output_postfix}.csv")
    # delete the original csv files
    for f in csv_files:
        os.remove(os.path.join(current_dir, f))
    else:
        print("No CSV files to concatenate")

# This is illustation codes. For the benefit of our viewers, we have uploaded an integrated file.

# current_dir is os.getcwd() + "/data". You can change it to your own path.
current_dir = os.getcwd() + "/raw data" 

prefix = "..." # I want to transform all the files that start with this prefix
convert_eprime_txt(prefix, current_dir) # Replace this with the prefix of your input files. Here I use the default output_postfix value wich is "_BehavioralRaw". You can change it by adding output_postfix="your_postfix" to the function call.

# endregion 2. Behavioral Data Transformation



# region 3. Behavioral Data Preprocessing (One merged raw date -> Cleaned Behavioral Data & Model Input)

import os
import pandas as pd
import numpy as np

# set the working directory

# Load behavior data and select/reorder columns
bhvr_df = pd.read_csv('data/eprime raw data.csv') # Load the behavior data
bhvr_df = bhvr_df.loc[bhvr_df['Procedure'] == 'TrialProc'] # Select only the rows with the "TrialProc" procedure
bhvr_df = bhvr_df.dropna(subset=['ActionClip.RESP']) # Drop rows with missing responses
bhvr_df = bhvr_df[['Subject', 'Group', 'Session', 'Correct', 'ClipName', 'ActionClip.ACC', 'ActionClip.RESP', 'ActionClip.RT']] # Select and reorder columns
# Rename columns
bhvr_df = bhvr_df.rename(columns={"Subject": "SUBJECT_ID", 
                                  "ClipName": "VIDEO_NAME",
                                  "Group": "SUBJECT_GROUP",
                                  "Correct": "OUTCOME",
                                  "ActionClip.RESP": "SUBJECT_RESPONSE",
                                  "Session": "CONDITION",
                                  "ActionClip.ACC": "ACCURACY",
                                  "ActionClip.RT": "RESPONSE_TIME"})
# bhvr_df['VIDEO_NAME'] = bhvr_df['VIDEO_NAME']# Extract the video name from the full video path
# Replace the "j" and "f" values with "2" and "1" respectively.This is to make the data compatible with the model. 
# "1" means left and "2" means right here.
bhvr_df['OUTCOME'] = bhvr_df['OUTCOME'].str.replace('j', '2').str.replace('f', '1') 
bhvr_df['SUBJECT_RESPONSE'] = bhvr_df['SUBJECT_RESPONSE'].str.replace('j', '2').str.replace('f', '1')

# remove invalid subjects 117(not qualified athlete level), 202, 203, 204, 205(pre-experiment subjects), 221(soccer player)
bhvr_df = bhvr_df[~bhvr_df['SUBJECT_ID'].isin([117, 202, 203, 204, 205, 221])]

# ordered the rows by subject id and condition
bhvr_df = bhvr_df.sort_values(by=['SUBJECT_ID', 'CONDITION'])
# remove the "resources/stim_" prefix and ".avi" suffix from the video names
bhvr_df['VIDEO_NAME'] = bhvr_df['VIDEO_NAME'].str.replace('resources/stim_', '').str.replace('.avi', '')


# Data Cleaning
all_df = bhvr_df[bhvr_df['RESPONSE_TIME'] >= 300] # Remove trials with response time < 300 ms
filtered_df = all_df[(all_df.groupby(['SUBJECT_ID', 'CONDITION'])['RESPONSE_TIME'].apply(lambda x: np.abs(x - x.mean()) <= 3 * x.std()))] # 保留3倍标准差内的数据 remove trials with response time > 3 standard deviations from the mean

# 计算保留试次数量、原有试次数量、保留率 Calculate the number of valid trials, the number of original trials, and the retention rate
results_df = pd.DataFrame(columns=['SUBJECT_ID', 'CONDITION', 'Original_Trials', 'Valid_Trials', 'Retention_Rate'])

for sub_con, sub_df in filtered_df.groupby(['SUBJECT_ID', 'CONDITION']):
    original_trials = len(bhvr_df[(bhvr_df['SUBJECT_ID'] == sub_con[0]) & (bhvr_df['CONDITION'] == sub_con[1])])
    clean_trials = len(sub_df)
    retained_trials = clean_trials / 120 # original_trials
    
    results_df = results_df.append({'SUBJECT_ID': sub_con[0], 'CONDITION': sub_con[1], 'Original_Trials': original_trials, 'Valid_Trials': clean_trials, 'Retention_Rate': retained_trials}, ignore_index=True)

# Save the data cleaning result
results_df.to_excel('data leaning result.xlsx', index=False)
# Save cleaned behavior data
filtered_df.to_csv('cleaned behaviroal data.csv', index=False) # Change the file name to your own


# Save model input data
exec_data = pd.read_excel('model input openpose.xlsx', sheet_name='Execution') # Load the execution data, we  create this file in step 1
with pd.ExcelWriter('model input.xlsx') as writer: # Save both dataframes in a single Excel file
    exec_data.to_excel(writer, sheet_name='Execution', index=False)
    filtered_df.to_excel(writer, sheet_name='Observation', index=False)

# endregion.



# region 4. Statisitcs Data Preparation(Model Output -> Statistic Data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function Definitions
# These functions allow for the computation of two indices of intersection between encoding and readout models: overlap and alignment. The compute_overlap function takes as input the weight vectors of the encoding and readout models and returns a scalar value in the range [0,1] representing the degree of overlap between the two vectors. The compute_alignment function takes the same input and returns a scalar value in the range [-1,1] representing the degree of alignment between the two vectors, taking into account both the magnitude and the sign of the coefficients.

def overlap(beta_enc, beta_read):
    """
    Computes the overlap index between encoding and readout weight vectors.

    Parameters
    ----------
    beta_enc : numpy array
        The weight vector for the encoding model.
    beta_read : numpy array
        The weight vector for the readout model.

    Returns
    -------
    float
        The overlap index, which is a value between 0 and 1.
    """
    dot_product = np.dot(np.abs(beta_enc), np.abs(beta_read))
    norm_product = np.linalg.norm(beta_enc) * np.linalg.norm(beta_read)
    return dot_product / norm_product

def alignment(beta_enc, beta_read):
    """
    Computes the alignment index between encoding and readout weight vectors.

    Parameters
    ----------
    beta_enc : numpy array
        The weight vector for the encoding model.
    beta_read : numpy array
        The weight vector for the readout model.

    Returns
    -------
    float
        The alignment index, which is a value between -1 and 1.
    """
    dot_product = np.dot(beta_enc, beta_read)
    norm_product = np.linalg.norm(beta_enc) * np.linalg.norm(beta_read)
    return dot_product / norm_product


# read pre-experiment data
all_df = pd.read_csv('cleaned behaviroal data.csv')

# 1. clean the behavioral data at the subject level 

mean_rt_raw = all_df.groupby(['SUBJECT_ID', 'SUBJECT_GROUP', 'CONDITION'])['RESPONSE_TIME'].mean().reset_index()

# 离群值检测
# 计算1.5倍四分位差
def calculate_outlier_bounds(values):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    return (q1 - 1.5*iqr, q3 + 1.5*iqr)

# 根据条件过滤数据并计算离群值
def find_outliers(df, group_col, condition_col, value_col):
    outliers = []
    for group in df[group_col].unique():
        for condition in df[condition_col].unique():
            subset = df[(df[group_col]==group) & (df[condition_col]==condition)]
            lower, upper = calculate_outlier_bounds(subset[value_col])
            condition_outliers = subset[(subset[value_col]<lower) | (subset[value_col]>upper)]
            outliers.append(condition_outliers['SUBJECT_ID'])
    return pd.concat(outliers)

# 如果暂时不想过滤离群值，可以直接使用原始数据
df_v = all_df

# 2. Format the data for analysis
mean_accuracy = df_v.groupby(['SUBJECT_ID', 'SUBJECT_GROUP', 'CONDITION'])['ACCURACY'].mean().reset_index()
# # compute the mean reaction time of each subject per condition
mean_rt = df_v.groupby(['SUBJECT_ID', 'SUBJECT_GROUP', 'CONDITION'])['RESPONSE_TIME'].mean().reset_index()

# compute the mean response bias of each subject per condition, which is the mean of the SUBJECT_RESPONSE of each trial, and then the result minus 1. 0.5 means no bias, 0 means bias to the left, 1 means bias to the right.
mean_bias = df_v.groupby(['SUBJECT_ID', 'SUBJECT_GROUP', 'CONDITION'])['SUBJECT_RESPONSE'].mean().reset_index()
mean_bias['SUBJECT_RESPONSE'] = mean_bias['SUBJECT_RESPONSE'] - 1

# merge these mean_accuracy, mean_rt and mean_bias using the subject ID, group, and condition
mean_df = mean_accuracy.merge(mean_rt, on=['SUBJECT_ID', 'SUBJECT_GROUP', 'CONDITION']).merge(mean_bias, on=['SUBJECT_ID', 'SUBJECT_GROUP', 'CONDITION'])
# rename the subject_response column to RESPONSE_BIAS
mean_df = mean_df.rename(columns={'SUBJECT_RESPONSE': 'RESPONSE_BIAS'})


# 3. Compute the overlap and alignment indices

# Notice: the raw model ouput is 
# compute overlap and alignment indices
enc_outs_resample = np.load('enc_outs_resample_PLD.npy', allow_pickle = True).item()
# mean value of encoding model coeffients, as the iput to calculate the overlap and alignment
enc_betas = np.mean(enc_outs_resample['betas'], axis=0) 

# 1. resampled_model x = (1, 1, 11, 200, False, False, 1)
# 2. permutation_model x = (1, 1, 11, 1, True, False, 1)
# 3. cross_validation_model x = (1, 1, 11, 1, False, True, 2)
# iC, iG: 11,21,12,22
model_results = np.load('readout_model_results.npy', allow_pickle = True)

readout_betas_G1C1 = np.mean(model_results[0]['betas'], axis=1)
readout_betas_G1C2 = np.mean(model_results[1]['betas'], axis=1)
readout_betas_G2C1 = np.mean(model_results[2]['betas'], axis=1)
readout_betas_G2C2 = np.mean(model_results[3]['betas'], axis=1)


def compute_indices(readout_betas, enc_betas):
    nsub = readout_betas.shape[0]
    overlap_indices = np.zeros((nsub, 1))
    alignment_indices = np.zeros((nsub, 1))
    for i in range(nsub):
        overlap_indices[i] = overlap(readout_betas[i], enc_betas)
        alignment_indices[i] = alignment(readout_betas[i], enc_betas)
    return overlap_indices, alignment_indices

overlap_indices_G1C1, alignment_indices_G1C1 = compute_indices(readout_betas_G1C1, enc_betas)
overlap_indices_G1C2, alignment_indices_G1C2 = compute_indices(readout_betas_G1C2, enc_betas)
overlap_indices_G2C1, alignment_indices_G2C1 = compute_indices(readout_betas_G2C1, enc_betas)
overlap_indices_G2C2, alignment_indices_G2C2 = compute_indices(readout_betas_G2C2, enc_betas)

# Readout CV accuracy
cvacc_G1C1 = model_results[8]['trial_acc']
cvacc_G1C2 = model_results[9]['trial_acc']
cvacc_G2C1 = model_results[10]['trial_acc']
cvacc_G2C2 = model_results[11]['trial_acc']

accuracies = []

for test in [cvacc_G1C1, cvacc_G1C2, cvacc_G2C1, cvacc_G2C2]:
    test_accuracies = []
    for person_results in test:
        person_accuracy = []
        for fold_results in person_results:
            fold_accuracy = np.mean(fold_results)
            person_accuracy.append(fold_accuracy)
        person_average_accuracy = np.mean(person_accuracy)
        test_accuracies.append(person_average_accuracy)
    accuracies.append(test_accuracies)

# copy the mean_df
df = mean_df.copy()

# sort the data by subject ID, group, and condition
df = df.sort_values(by=['CONDITION','SUBJECT_GROUP', 'SUBJECT_ID'])

df['OVERLAP'] = np.concatenate((overlap_indices_G1C1, overlap_indices_G2C1, overlap_indices_G1C2, overlap_indices_G2C2), axis=0)
df['ALIGNMENT'] = np.concatenate((alignment_indices_G1C1, alignment_indices_G2C1, alignment_indices_G1C2, alignment_indices_G2C2), axis=0)

df['READOUT'] = np.concatenate((accuracies[0], accuracies[2],accuracies[1], accuracies[3]), axis=0)

# sort df by index
df = df.sort_index()


# endregion.



# region 5. Corrs-tasks Overlap & Alinment

# define functions
def compute_indices(readout_betas, enc_betas, nsub):
    nsub = nsub
    overlap_indices = np.zeros((nsub, 1))
    alignment_indices = np.zeros((nsub, 1))
    for i in range(nsub):
        overlap_indices[i] = overlap(readout_betas[i], enc_betas[i])
        alignment_indices[i] = alignment(readout_betas[i], enc_betas[i])
    return overlap_indices, alignment_indices

# load the encoding results
enc_outputs = np.load('encoding_results.npy', allow_pickle = True)
enc_betas = enc_outputs[0]['betas']
enc_betas = np.mean(enc_betas, axis=0)
# sort enc_betas using absolute values in descending order and get the index
enc_betas_sorted = np.argsort(np.abs(enc_betas))[::-1]

# load the readout results
readout_model_results = np.load('readout_model_results.npy', allow_pickle = True) # load the results of the readout model
# 0-3, resampled model; 4-7, permutation model; 8-11, cross-validation model
# Order: Expert-NVs, Expert-PLDs, Novice-NVs, Novice-PLDs
from utils import compute_pvalues # import the function to compute p-values

nsub = 18

# Comparison 1: Expert-NVs vs Expert-PLDs
readbetas_Expert_ND = readout_model_results[0]['betas'].mean(axis=1) # Expert-NVs; n subjects x n features(18 subjects * 22 features)
readbetas_Expert_PLD = readout_model_results[1]['betas'].mean(axis=1) # Expert-PLDs; n subjects x n features
overlap_Experts_ND_PLD, alignment_Experts_ND_PLD = compute_indices(readbetas_Expert_ND, readbetas_Expert_PLD, nsub)

# Comparison 2: Novice-NDs vs Novice-PLDs
readbetas_Novice_ND = readout_model_results[2]['betas'].mean(axis=1) # Novice-NVs; n subjects x n features
readbeta_Novice_PLD = readout_model_results[3]['betas'].mean(axis=1) # Novice-PLDs; n subjects x n features
overlap_Novice_ND_PLD, alignment_Novice_ND_PLD = compute_indices(readbetas_Novice_ND, readbeta_Novice_PLD, nsub)


# endregion.

