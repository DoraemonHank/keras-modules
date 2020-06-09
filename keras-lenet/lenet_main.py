import sys
# sys.path.append("./keras-lenet")
from lenet import Lenet,Lenet2

# from resnets import ResNet50
from module_utils import load_dataset, preprocess_data, plot_model_history,create_dataset

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.optimizers import SGD


# create_dataset('./pet_data','training_set',64,64,2)
# create_dataset('./pet_data','test_set',64,64,2)
    

# load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset('./pet_data','training_set.h5','test_set.h5')

# preprocess dataset
X_train, Y_train, X_test, Y_test = preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig,len(classes))

# generate model
model = Lenet(64,64,3,classes = len(classes))
#model = ResNet50(input_shape = (64, 64, 3), classes = len(classes))

# compile modelimport sys

# sys.path.append("./keras-signs-resnet")
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
model_history = model.fit(X_train, Y_train,epochs = 100,shuffle=True, batch_size = 32)

# plot accuracy/loss
plot_model_history(model_history,'accuracy')

# evaluate model
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# single prediction
from keras.preprocessing import image
import numpy as np
test_image = image.load_img('./pet_data/test_set/dogs/dog.4017.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result[0][0] > result[0][1]:
    prediction = 'cat'
else:
    prediction = 'dog'
print(prediction)

# Show image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img = mpimg.imread('./pet_data/single_prediction/cat_or_dog_2.jpg')
plt.imshow(img)

