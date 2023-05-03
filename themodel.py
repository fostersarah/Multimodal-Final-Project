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

print(XData[i])
print(YData)
