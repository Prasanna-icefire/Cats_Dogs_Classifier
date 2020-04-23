import tensorflow as tf 
import numpy as np 
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D, Activation

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
#We need to always normalize the data before we feed into model. We need to scale our model first.
#For gray scale images 0 represents black wile 255 is white.....So we need to normalize all the pixel values, convert every pixel in the range 0 to 1
X=X/255.0
print(X[1])
model = Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(np.array(X),np.array(y), batch_size=32,epochs = 10, validation_split=0.1)

