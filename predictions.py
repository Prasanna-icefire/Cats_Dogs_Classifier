import tensorflow as tf 
import cv2
CATEGORIES = ["Dog","Cat"]
IMAGE_SIZE=60
def prepare(filepath):
    IMAGE_SIZE = 60
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
    return new_array.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
model = tf.keras.models.load_model("Dog_CAT_CNN.model")
prediction = model.predict([prepare("dog.jpg")])
print(CATEGORIES[int(prediction[0][0])])

