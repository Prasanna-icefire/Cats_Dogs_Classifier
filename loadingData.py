import cv2 as cv
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import random
DATADIR = "/home/icefire/ML/Cats_vs_Dogs/Data_Set/PetImages"
CATEGORIES = ["Dog","Cat"]
training_data = []
IMG_SIZE = 60
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
                newArray = cv.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([newArray,class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))
random.shuffle(training_data)

X = []
y = []

for features,labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
print(X[1])



