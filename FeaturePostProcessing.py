import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler

## Creates final csv files if they don't exist
with open('VideoFeatures/landmarks.csv', 'w') as csvfile:
        csvfile.close()
        
with open('VideoFeatures/AU1.csv', 'w') as csvfile:
        csvfile.close()
        
with open('VideoFeatures/AU2.csv', 'w') as csvfile:
        csvfile.close()
        
landmarks = pd.DataFrame()
AU1 = pd.DataFrame()
AU2 = pd.DataFrame()

## Loops over every data point to process
for x in range (1, 77, 1):

    ## Grab the current data points extracted features
    inputFile = "RawFeatures/" + str(x) + ".csv"
    inputDF = pd.read_csv(inputFile)
    
    ## split into sets of features
    
    ## Facial Landmarks
    features1 = inputDF.loc[:, " x_0":" y_67"]
    
    ## Action Units (Intensity)
    features2 = inputDF.loc[:, " AU01_r":" AU45_r"]
    
    ## Action Units (Presence)
    features3 = inputDF.loc[:, " AU01_c":" AU45_c"]

    ## For each Feature set, add rows of zero until divisible by zero
    num = 10 - features1.index.size % 10 

    for i in range(num):
        features1 = pd.concat([features1, pd.Series(0, index=features1.columns).to_frame().T], ignore_index=True)
        features2 = pd.concat([features2, pd.Series(0, index=features2.columns).to_frame().T], ignore_index=True)    
        features3 = pd.concat([features3, pd.Series(0, index=features3.columns).to_frame().T], ignore_index=True)
    
    
    
    # 1. Facial Landmarks
    
    ## array to store linear feature vector
    concat = np.empty(int(features1.index.size/10) * features1.columns.size)
    index = 0
    for i in range (0, features1.index.size - 9, 10): ## moves down the rows, in groups of ten
        for j in range (features1.columns.size): ## moves across the columns
            concat[index] = features1.iloc[i:i+10, j:j+1].mean() ## Average every ten frames together
            index += 1
            
    ## Append to dataframe
    
    data = pd.Series(concat).to_frame().T 
    landmarks = pd.concat([landmarks, data], ignore_index=True)
             
    
    # 2. Action Units (Intensity)
    
    
    concat = np.empty(int(features2.index.size/10) * features2.columns.size)
    index = 0
    for i in range (0, features2.index.size - 9, 10): 
        for j in range (features2.columns.size): 
            concat[index] = features2.iloc[i:i+10, j:j+1].mean() 
            index += 1
            
    # Append to dataframe
    
    data = pd.Series(concat).to_frame().T 
    AU1 = pd.concat([AU1, data], ignore_index=True)
    
    # 3. Action Units (Presence)
    
    
    concat = np.empty(int(features3.index.size/10) * features3.columns.size)
    index = 0
    for i in range (0, features3.index.size - 9, 10): 
        for j in range (features3.columns.size): 
            concat[index] = features3.iloc[i:i+10, j:j+1].mean() 
            index += 1
    
    ## Append to dataframe
    
    data = pd.Series(concat).to_frame().T 
    AU2 = pd.concat([AU2, data], ignore_index=True)

## sklearn scaling on non-NaN data
scaler = MinMaxScaler((-1, 1))

for column in landmarks.columns:
    null_index = landmarks[column].isnull()
    landmarks.loc[~null_index, [column]] = scaler.fit_transform(landmarks.loc[~null_index, [column]])

for column in AU1.columns:
    null_index = AU1[column].isnull()
    AU1.loc[~null_index, [column]] = scaler.fit_transform(AU1.loc[~null_index, [column]])

for column in AU2.columns:
    null_index = AU2[column].isnull()
    AU2.loc[~null_index, [column]] = scaler.fit_transform(AU2.loc[~null_index, [column]])

## NaN values represent vectors of smaller length than the largest, feature pad with zeros.
landmarks.fillna(0.0, inplace = True)
AU1.fillna(0.0, inplace = True)
AU2.fillna(0.0, inplace = True)

## Fix indexing
landmarks.index = np.arange(1, len(landmarks) + 1)
AU1.index = np.arange(1, len(AU1) + 1)
AU2.index = np.arange(1, len(AU2) + 1)

## Send data to csv files
landmarks.to_csv('VideoFeatures/landmarks.csv')
AU1.to_csv('VideoFeatures/AU1.csv')
AU2.to_csv('VideoFeatures/AU2.csv')