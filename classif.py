import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
import skimage as sk
from skimage import io
from skimage import transform

from sklearn import metrics as ms
from sklearn import neural_network
from sklearn import ensemble
from sklearn import tree
from sklearn import svm


from sklearn.cross_validation import train_test_split
from PIL import Image
from matplotlib.pyplot import imshow
from sklearn.preprocessing import OneHotEncoder

def splitTrainTest(df, rate, minLength):
    randomIndex = np.arange(minLength)
    np.random.shuffle(randomIndex)
    randomIndex = randomIndex.tolist()
    splitPlace = int(np.floor(len(df)*rate)) # min 1 test beacuse of floor
    train = df.iloc[randomIndex[:splitPlace]]
    test = df.iloc[randomIndex[splitPlace:]]

    return train, test

def splitTrainTestEqual(df, target, rate):
    labels = list(set(df[target]))

    testList = []
    trainList = []

    minLength = 1e28
    for l in labels:
        tmpDf = df[df[target] == l]
        if minLength > len(tmpDf):
            minLength = len(tmpDf)

    for l in labels:
        tmpDf = df[df[target] == l]
        tmpTrain, tmpTest = splitTrainTest(tmpDf, rate, minLength)
        testList.append(tmpTest)
        trainList.append(tmpTrain)

    test = pd.concat(testList, ignore_index = True)
    train = pd.concat(trainList, ignore_index = True)

    test = pd.DataFrame(test.values, columns = df.columns)
    train = pd.DataFrame(train.values, columns = df.columns)

    return train, test

def classification(df, features, target, clf, iteration = 1, rate = 0.8):
    precision = 0

    for i in range(iteration):
        train, test = splitTrainTestEqual(df, target, rate)

        clf.fit( train[features].values , train[target].values)

        predictions = clf.predict(test[features].values)
        precision += ms.accuracy_score(test[target].values,predictions)

    precision /= iteration

    return precision, clf


#clf = neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(20,), max_iter = 500)
#clf = svm.SVC(gamma=0.01)
clf = ensemble.RandomForestClassifier(n_estimators=2000)
precision, clf = classification(imgDf, features, target, clf)

print(precision)

predictions = clf.predict(testDf[features].values)
testDf[target] = predictions

output = testDf[[ID, target]]
output.to_csv('beHappy.csv', index = False)
print(df.head())
