import numpy
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

AU1file = "VideoFeatures/AU1.csv"
AU2file = "VideoFeatures/AU2.csv"
heartRatefile = "PhysiologicalFeatures/physiologicalData.csv"
RPE= pd.read_csv("Aw-Soma Data.csv")
RPE = RPE['RPE'][:76] 

AU1 = pd.read_csv(AU1file);
AU2 = pd.read_csv(AU2file);
HR = pd.read_csv(heartRatefile);



XData= []

for i in range(0,75):
    XData.append([AU1.iloc[i],AU2.iloc[i],HR.iloc[i]])


YData = RPE

AU1data = "VideoFeatures/AU1.csv"
AU2data = "VideoFeatures/AU2.csv"
Landmarkdata = "VideoFeatures/landmarks.csv"
heartRateData = "PhysiologicalFeatures/physiologicalData.csv"

# Load Data
AU1 = pd.read_csv(AU1data)
AU1 = AU1.rename(columns = {'Unnamed: 0':'Index'})

AU2 = pd.read_csv(AU2data)
AU2 = AU2.rename(columns = {'Unnamed: 0':'Index'})

Landmarks = pd.read_csv(Landmarkdata)
Landmarks = Landmarks.rename(columns = {'Unnamed: 0':'Index'})

physData = pd.read_csv(heartRateData)
physData['Index'] = physData['Index'] + 1

FeatureSet1 = pd.merge(physData, AU1, how = "left", on = "Index")
FeatureSet2 = pd.merge(physData, AU2, how = "left", on = "Index")
FeatureSet3 = pd.merge(physData, Landmarks, how = "left", on = "Index")