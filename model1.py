from keras.src.legacy.preprocessing.image import ImageDataGenerator

data_augmentation = True

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

dataToTrain.drop('label', axis=1, inplace=True)

images = dataToTrain.values

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

images = images / 255

images = images.reshape(images.shape[0], 28, 28, 1)

if data_augmentation:
    # With data augmentation to prevent overfitting
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(images)

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.3, random_state=84)

batch_size = 128
num_classes = 24
epochs = 50

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=epochs,
                    batch_size=batch_size)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy graph")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])

plt.show()

model.save('model.keras')
