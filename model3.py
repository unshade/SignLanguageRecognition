""""This model is a CNN model based on the VGG-16 architecture for 28x28 images with data augmentation."""""

import pandas as pd
from keras import Sequential
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import numpy as np

dataToTrain = pd.read_csv('./data/sign_mnist_train/sign_mnist_train.csv')

labels = dataToTrain['label'].values
unique_labels = np.sort(pd.unique(labels))

dataToTrain.drop('label', axis=1, inplace=True)

images = dataToTrain.values

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

# Normalize the data
images = images / 255

# Reshaping the data from 1-D to 3-D for CNN input
images = images.reshape(-1, 28, 28, 1)

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

# Splitting the data into training and validation
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.3, random_state=84)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

numbers_of_classes = len(unique_labels)

model = Sequential()
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=numbers_of_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(datagen.flow(images_train, labels_train, batch_size=128), epochs=50, validation_data=(images_test,
                                                                                                          labels_test),
                    callbacks=[learning_rate_reduction])

print("Accuracy of the model is - ", model.evaluate(images_test, labels_test)[1] * 100, "%")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy graph")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])

plt.show()

# Save the model
model.save('model.keras')
