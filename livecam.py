import cv2
import numpy as np
from keras.models import load_model
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

alp = 'ABCDEFGHIKLMNOPQRSTUVWXY'

dataToTest = pd.read_csv('./data/sign_mnist_test/sign_mnist_test.csv')


test_labels = dataToTest['label']
unique = np.unique(test_labels)
unique = np.sort(unique)
combined = dict(zip(unique, alp))
print(combined)
dataToTest.drop('label', axis = 1, inplace = True)
test_images = dataToTest.values
label_binrizer = LabelBinarizer()
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

model = keras.models.load_model('./model.h5')

# Initialiser la caméra
cap = cv2.VideoCapture(2)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # define region of interest
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imshow('roi sacled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

    roi = roi.reshape(1, 28, 28, 1)

    #result = model.predict_classes(roi, 1, verbose=0)[0]
    result = str(np.argmax(model.predict(roi), axis=-1)[0])
    result = int(result)
    result = unique[result]
    result = combined[int(result)]
    cv2.putText(copy, result, (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

