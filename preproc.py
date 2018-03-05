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

def getImageAsVector(image):
    imgArray = sk.io.imread(image) # load as matrix
    imgArray = sk.color.rgb2grey(imgArray) # convert to gray
    
    imgShape = imgArray.shape
    imgArray = imgArray.reshape(imgShape[0]*imgShape[1]) # convert to vector format
    
    return list(imgArray)


def matrixToDataframe(data, colNames = None):
    if colNames is None:
        colNames = np.arange(data.shape[1]).astype(str)

    df = pd.DataFrame( data, columns = colNames)

    return df


def images2Dataframe(labelFiel, folder):
    """
    Convert imgs in a folder to vectors. Add label to images according to the label file.
    Return as a dataframe contain the vectors, and labels.
    """
    # Read the csv conatin img IDs and labes.
    df = pd.read_csv(labelFiel)
    cols = df.columns
    ID = cols[0]
    target = cols[1]
    imgList = list(df[ID])
    
    # Craet MAtrix with zeroes will conatin all images as a vecor
    numOfImgs = len(imgList) # get the num of images
    tmpImg = getImageAsVector(folder + str(imgList[1]) + '.png')
    size = len(tmpImg) # Get the img vecot size
    M = np.zeros([numOfImgs, size]) # Matrix contain all image

    # Add values vector values to matrix
    for i in range(numOfImgs):
        imgName = folder + str(imgList[i]) + '.png'
        M[i,:] = getImageAsVector(imgName)
        
    imgDf = matrixToDataframe(M) # convert matrix to dataframe
    features = imgDf.columns
    imgDf[cols] = df[cols] # add label and ID to image dataframe   
    
    return imgDf, features, target, ID




imgDf, features, target, ID = images2Dataframe('train_labels.csv', 'trainData/')
imgDf.to_csv('usedTrainData.csv', index=False)

testDf, features, target, ID = images2Dataframe('beHappy.csv', 'testData/')
testDf.to_csv('usedTestData.csv', index=False)


# IT just for fun