# Cats_Dogs_Classifier
This is a repository consisting of a model I have created to distinguish between a dog and a cat.

Prerequisites:
python
tensorflow   installed by "pip install tensorflow"   I used version 2.0 with gpu comatibility,how ever GPU is not that necessary
opencv       installed by "pip install opencv-python" or "pip install opencv-contrib-python"

To use this as a direct application,create a folder in which you want to use this.
Inside the folder, go ahead and download the DOG_CAT_CNN.model and the predictions.py python file.
In the python file,change the location of the variable pic,to the location wherever the picture is located and run the file.
The result is obtained on the command_line.

For those who want to understand how the model is created,how data set is created and such :
go ahead and download a set of images of dogs and a set of images of cats. 
Microsoft has done this easy by providing with about 12000 images of each,cats and dogs. To use this,you can download the images.
Link "https://www.youtube.com/redirect?redir_token=62lpFwOcr9iYHfSq78JvuGC4sot8MTU4ODAwODI4OEAxNTg3OTIxODg4&event=video_description&v=j-3vuBynnOE&q=https%3A%2F%2Fwww.microsoft.com%2Fen-us%2Fdownload%2Fconfirmation.aspx%3Fid%3D54765"

Having got these images, we need to convert the images into grayscale and a numpy array to have it saved in the form of numbers.
Before doing so,it is necessary to scale all the images to the same size.This is done using cv2.resize().All this is understandable from loadingData.py

Having obtained this data in the form that is feedable into the model we create,we can write the create_model.py.
I have used a Sequential() to define the model. I have used numerous layers of 2D Convolution layers,dense and flattened layers.
The code is available and is quiet understandable. Having performed a numerous tests, I managed to get validation accuracies and test accuracies around 83%.
The model is thus saved and is used in the prediction.py file.

Happy Coding!

