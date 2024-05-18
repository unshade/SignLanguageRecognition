import pandas as pd
import numpy as np

dataToTrain = pd.read_csv('./data/sign_mnist_train/sign_mnist_train.csv')
dataToTest = pd.read_csv('./data/sign_mnist_test/sign_mnist_test.csv')
alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'

print("-- DATA TO BE TRAINED --")
#print(dataToTrain)
print("-- DATA TO BE TESTED --")
#print(dataToTest)


print("-- POSSIBLE UNIQUE LABELS --")
labels = dataToTrain['label'].values
unique_labels = np.sort(pd.unique(labels))
mapping = dict(zip(unique_labels, alphabet))
print(mapping)

# Prepare data for training
# Drop the label column from the training data
trainingImages = dataToTrain.drop('label', axis=1).values
# Each image is 28x28 pixels
trainingImages = [np.reshape(image, (28, 28)) for image in trainingImages]