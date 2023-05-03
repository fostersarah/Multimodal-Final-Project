import numpy
import pandas as pd
from sklearn import svm

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