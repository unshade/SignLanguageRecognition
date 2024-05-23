import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score

dataToTest = pd.read_csv('./data/sign_mnist_test/sign_mnist_test.csv')

test_labels = dataToTest['label']
dataToTest.drop('label', axis=1, inplace=True)
test_images = dataToTest.values
label_binrizer = LabelBinarizer()
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

model = keras.models.load_model('./model.keras')

y_pred = model.predict(test_images)

print("Accuracy: ", accuracy_score(test_labels, y_pred.round()))

print("Precision: ", precision_score(test_labels, y_pred.round(), average='weighted'))
print("Recall: ", recall_score(test_labels, y_pred.round(), average='weighted'))
