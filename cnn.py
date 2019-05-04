# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,( 3, 3), input_shape = (3,64,64), activation = 'relu',data_format="channels_first"))

# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu',data_format="channels_first"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128,kernel_initializer ='uniform', activation = 'relu'))
classifier.add(Dropout(.2))
classifier.add(Dense(output_dim = 64, kernel_initializer ='uniform',activation = 'relu'))
classifier.add(Dropout(.2))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,data_format="channels_first")

test_datagen = ImageDataGenerator(rescale=1./255,data_format="channels_first")

train_generator = train_datagen.flow_from_directory(
        '/Users/jasdeepsingh/Downloads/Convolutional_Neural_Networks/dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/Users/jasdeepsingh/Downloads/Convolutional_Neural_Networks/dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,  
        epochs=10,
        validation_data=validation_generator,
        validation_steps=2000)

classifier_json =classifier.to_json()
with open("classifier.json","w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("classifier.h5")


#Loading the data
from keras.models import model_from_json
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifier.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#evaluate
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/jasdeepsingh/Downloads/Convolutional_Neural_Networks/dataset/white.jpeg',target_size=(64,64))
test_image = image.img_to_array(test_image,data_format="channels_first")
test_image = np.expand_dims(test_image,axis=0)
result = loaded_model.predict_proba(test_image)
if result == 1:
    print("This is a Dog Image")
else:
    print("This is a Cat Image")

