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

AU1 = pd.read_csv(AU1file);
AU2 = pd.read_csv(AU2file);
HR = pd.read_csv(heartRatefile);

print(AU1);
print(AU2)
print(HR)
#looks like all the data is given in the index for the participant
