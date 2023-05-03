import numpy
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

AU1file = "VideoFeatures/AU1.csv"
AU2file = "VideoFeatures/AU2.csv"
Landmarkfile = "VideoFeatures/landmarks.csv"
heartRatefile = "PhysiologicalFeatures/physiologicalData.csv"

# Load Data
AU1 = pd.read_csv(AU1file)
AU1 = AU1.rename(columns = {'Unnamed: 0':'Index'})

AU2 = pd.read_csv(AU2file)
AU2 = AU2.rename(columns = {'Unnamed: 0':'Index'})

Landmarks = pd.read_csv(Landmarkfile)
Landmarks = Landmarks.rename(columns = {'Unnamed: 0':'Index'})

physData = pd.read_csv(heartRatefile)
physData['Index'] = physData['Index'] + 1

FeatureSet1 = pd.merge(physData, AU1, how = "left", on = "Index")
FeatureSet2 = pd.merge(physData, AU2, how = "left", on = "Index")
FeatureSet3 = pd.merge(physData, Landmarks, how = "left", on = "Index")

RPE= pd.read_csv("Aw-Soma Data.csv")
RPE = RPE['RPE'][:76] 
YData = RPE

#randomly splitting the data
trainX, testX, trainY, testY = train_test_split(FeatureSet1, YData, test_size=0.3)

randomForest = RandomForestClassifier(n_estimators=40, max_features="sqrt")

randomForest.fit(trainX, trainY)

predictionF1 = randomForest.predict(testX)

rfF1Accuracy =accuracy_score(testY, predictionF1)

print(rfF1Accuracy)