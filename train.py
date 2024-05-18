import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




dataToTrain = pd.read_csv('./data/sign_mnist_train/sign_mnist_train.csv')

labels = dataToTrain['label'].values
unique_labels = np.sort(pd.unique(labels))

dataToTrain.drop('label', axis = 1, inplace = True)

images = dataToTrain.values

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size = 0.2, random_state = 84)

batch_size = 128
num_classes = 24
epochs = 50

images_train = images_train / 255
images_test = images_test / 255

images_train = images_train.reshape(images_train.shape[0], 28, 28, 1)
images_test = images_test.reshape(images_test.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(images_train, labels_train, validation_data = (images_test, labels_test), epochs=epochs, batch_size=batch_size)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy graph")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()

model.save('model.h5')