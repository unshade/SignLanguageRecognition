import pandas as pd

dataToTrain = pd.read_csv('./data/sign_mnist_train/sign_mnist_train.csv')
dataToTest = pd.read_csv('./data/sign_mnist_test/sign_mnist_test.csv')

print("-- THIS IS THE DATA TO BE TRAINED --")
print(dataToTrain)
print("-- THIS IS THE DATA TO BE TESTED --")
print(dataToTest)
