import numpy
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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

FeatureSet1 = FeatureSet1.drop('Index', axis=1)
FeatureSet2 = FeatureSet2.drop('Index', axis=1)
FeatureSet3 = FeatureSet3.drop('Index', axis=1)

RPE= pd.read_csv("Aw-Soma Data.csv")
RPE = RPE['RPE'][:76] 
RPE = RPE.replace([1,2,3,4], 4)
RPE = RPE.replace([5,6,7], 7)
RPE = RPE.replace([8,9,10], 10)

YData = RPE

#randomly splitting the data for feature 1 and testing
trainX, testX, trainY, testY = train_test_split(FeatureSet1, YData, test_size=0.3)

randomForest = RandomForestClassifier(n_estimators=40, max_features="sqrt")

randomForest.fit(trainX, trainY)

predictionF1 = randomForest.predict(testX)

rfF1Accuracy =accuracy_score(testY, predictionF1)


print("Accuracy score for AU1 and heart rate as x: " + str(rfF1Accuracy))


#featureset2 as the x here
trainX, testX, trainY, testY = train_test_split(FeatureSet2, YData, test_size=0.3)

randomForest = RandomForestClassifier(n_estimators=40, max_features="sqrt")

randomForest.fit(trainX, trainY)

predictionF2 = randomForest.predict(testX)

rfF2Accuracy =accuracy_score(testY, predictionF2)

print("Accuracy score for AU2 and heart rate as x: " + str(rfF2Accuracy))

#featureset3 as the x here
trainX, testX, trainY, testY = train_test_split(FeatureSet3, YData, test_size=0.3)

randomForest = RandomForestClassifier(n_estimators=40, max_features="sqrt")

randomForest.fit(trainX, trainY)

predictionF3 = randomForest.predict(testX)

rfF3Accuracy =accuracy_score(testY, predictionF3)

print("Accuracy score for landmarks and heart rate as x: " + str(rfF3Accuracy))


#only heart rate as the x here
trainX, testX, trainY, testY = train_test_split(physData, YData, test_size=0.3)

randomForest = RandomForestClassifier(n_estimators=40, max_features="sqrt")

randomForest.fit(trainX, trainY)

predictionHR = randomForest.predict(testX)

rfHRAccuracy =accuracy_score(testY, predictionHR)

print("Accuracy score for just heart rate as x: " + str(rfHRAccuracy))


## Using GridSearchCV to find best parameter lists

####### Random Forest #######

# building base models to search with

trainX1, testX1, trainY1, testY1 = train_test_split(FeatureSet1, YData, test_size=0.3)
randomForest1 = RandomForestClassifier(n_estimators=40, max_features="sqrt")
randomForest.fit(trainX1, trainY1)
predictionF1 = randomForest.predict(testX1)
rfF1Accuracy =accuracy_score(testY1, predictionF1)

print("Accuracy score for AU1 and heart rate as x (RF): " + str(rfF1Accuracy))

trainX2, testX2, trainY2, testY2 = train_test_split(FeatureSet2, YData, test_size=0.3)
randomForest2 = RandomForestClassifier(n_estimators=40, max_features="sqrt")
randomForest.fit(trainX2, trainY2)
predictionF2 = randomForest.predict(testX2)
rfF2Accuracy =accuracy_score(testY2, predictionF2)

print("Accuracy score for AU2 and heart rate as x (RF): " + str(rfF2Accuracy))

trainX3, testX3, trainY3, testY3 = train_test_split(FeatureSet3, YData, test_size=0.3)
randomForest3 = RandomForestClassifier(n_estimators=40, max_features="sqrt")
randomForest.fit(trainX3, trainY3)
predictionF3 = randomForest.predict(testX3)
rfF3Accuracy =accuracy_score(testY3, predictionF3)

print("Accuracy score for landmarks and heart rate as x (RF): " + str(rfF3Accuracy))

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid1 = GridSearchCV(estimator = randomForest1, param_grid=param_grid, refit = True, verbose = 3)
grid2 = GridSearchCV(estimator = randomForest2, param_grid=param_grid, refit = True, verbose = 3)
grid3 = GridSearchCV(estimator = randomForest3, param_grid=param_grid, refit = True, verbose = 3)

grid1.fit(trainX1, trainY1)
predictionF1 = grid1.predict(testX1)
rfF1Accuracy =accuracy_score(testY1, predictionF1)

print("Parameter search accuracy score for AU1 and heart rate as x (RF): " + str(rfF1Accuracy))
print(grid1.best_params_)
print(grid1.best_estimator_)

grid2.fit(trainX2, trainY2)
predictionF2 = grid2.predict(testX2)
rfF2Accuracy =accuracy_score(testY2, predictionF2)

print("Parameter search accuracy score for AU2 and heart rate as x (RF): " + str(rfF2Accuracy))
print(grid2.best_params_)
print(grid2.best_estimator_)

grid3.fit(trainX3, trainY3)
predictionF3 = grid3.predict(testX3)
rfF3Accuracy =accuracy_score(testY3, predictionF3)

print("Parameter search accuracy score for landmarks and heart rate as x (RF): " + str(rfF3Accuracy))
print(grid3.best_params_)
print(grid3.best_estimator_)

####### SVC #######

# building base models to search with

svc1 = SVC()
svc1.fit(trainX1, trainY1)
predictionF1 = svc1.predict(testX1)
svcF1Accuracy = accuracy_score(testY1, predictionF1)

print("Accuracy score for AU1 and heart rate as x (SVC): " + str(svcF1Accuracy))

svc2 = SVC()
svc2.fit(trainX2, trainY2)
predictionF2 = svc2.predict(testX2)
svcF2Accuracy =accuracy_score(testY2, predictionF2)

print("Accuracy score for AU2 and heart rate as x (SVC): " + str(svcF2Accuracy))

svc3 = SVC()
svc3.fit(trainX3, trainY3)
predictionF3 = svc3.predict(testX3)
svcF3Accuracy =accuracy_score(testY3, predictionF3)

print("Accuracy score for landmarks and heart rate as x (SVC): " + str(svcF3Accuracy))

 
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid1 = GridSearchCV(estimator = svc1, param_grid=param_grid, refit = True, verbose = 3)
grid2 = GridSearchCV(estimator = svc2, param_grid=param_grid, refit = True, verbose = 3)
grid3 = GridSearchCV(estimator = svc3, param_grid=param_grid, refit = True, verbose = 3)

grid1.fit(trainX1, trainY1)
predictionF1 = grid1.predict(testX1)
svcF1Accuracy =accuracy_score(testY1, predictionF1)

print("Parameter search accuracy score for AU1 and heart rate as x (SVC): " + str(svcF1Accuracy))
print(grid1.best_params_)
print(grid1.best_estimator_)

grid2.fit(trainX2, trainY2)
predictionF2 = grid2.predict(testX2)
svcF2Accuracy =accuracy_score(testY2, predictionF2)

print("Parameter search accuracy score for AU2 and heart rate as x (SVC): " + str(svcF2Accuracy))
print(grid2.best_params_)
print(grid2.best_estimator_)

grid3.fit(trainX3, trainY3)
predictionF3 = grid3.predict(testX3)
svcF3Accuracy =accuracy_score(testY3, predictionF3)

print("Parameter search accuracy score for landmarks and heart rate as x (SVC): " + str(svcF3Accuracy))
print(grid3.best_params_)
print(grid3.best_estimator_)