# importing required components
from turtle import pd
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import multiclass
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import ssl, os, time

# fetching data
X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)
print(pd.Series(y),pd.value_counts())
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

# splitting and scaling data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

# scaling them
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# fitting the data into the model
clf = LogisticRegression(solver='saga', multi_class= 'multinomial'.fit(X_train_scaled, y_train))

# calculationg accuracy
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is: ", accuracy)

# starting the camera
capture = cv2.VideoCapture(0)

while(True):
    # capturing frame-by-frame
    try:
        ret, frame = capture.read()

        #  operations on frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Drawing a box in the centre of the video
        height, width = gray.shape()
        upper_left = (int(width/2-68), int(height/2-68))
        bottom_right = (int(width/2+68), int(height/2+68))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        # to consider the area within the box
        # roi = region of interest
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        # converting cv2 - pil format
        im_pil = Image.formarray(roi)

        # convert to grayscale image - 'L' format means each pixel is 
        # represented by a single value from 0 to 255
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize(28, 28), Image.ANTIALIAS
        
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)

        # display resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass


# release cpature when everything is done
capture.release()
cv2.destroyAllWindows()


