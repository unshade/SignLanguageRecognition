import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
dataToTrain = pd.read_csv('./data/sign_mnist_train/sign_mnist_train.csv')
dataToTest = pd.read_csv('./data/sign_mnist_test/sign_mnist_test.csv')
alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'

#print(dataToTrain)
#print(dataToTest)


labels = dataToTrain['label'].values
unique_labels = np.sort(pd.unique(labels))
mapping = dict(zip(unique_labels, alphabet))

# Prepare data for training
# Drop the label column from the training data
trainingImages = dataToTrain.drop('label', axis=1).values

# Binarize the labels
labels = LabelBinarizer().fit_transform(labels)