# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:07:41 2020

@author: Hank
"""

import os
import cv2
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

import shutil
import random 

path_train = "E:\\ocr\\train"
path_test = "E:\\ocr\\test"
dirs = os.listdir( path_train )
file_buf = []
file2_buf = []

for file in dirs:
    
    file2_buf = []
    dirs2 = os.listdir( path_train + '\\' + file)
    for file2 in dirs2:  
        # print(file2)
        if file2[-3:] == 'jpg' or file2[-4:] == 'jpeg' :
            file2_buf.append(file2)
    file_buf.append(file2_buf)


dirs = os.listdir( path_test )   

for file_index in range(len(dirs)):
    print(dirs[file_index])
    randomIndex = random.sample(range(file_buf[file_index].__len__()  ), 10)  
    for i in randomIndex: 
        print (file_buf[file_index][i])
        shutil.move(path_train + '\\' + dirs[file_index] + '\\' + file_buf[file_index][i],path_test + '\\' + dirs[file_index] + '\\')
    


# def loadData(path):
#     data = []
#     labels = []
#     dirs = os.listdir( path )
#     for i in range(len(dirs)):
#         dir = path + '\\' + dirs[file_index]
#         listImg = os.listdir(dir)
#         for img in listImg:
#             data.append([cv2.imread(dir+'/'+img, 0)])
#             labels.append(i)
#         print(path, i, 'is read')
#     return data, labels


# trainData, trainLabels = loadData(path_train)
# testData, testLabels = loadData(path_test)
# trainLabels = np_utils.to_categorical(trainLabels, 36)
# testLabels = np_utils.to_categorical(testLabels, 36)


model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(36, activation='softmax'))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(trainData, trainLabels, batch_size=500, epochs=20, verbose=1, shuffle=True)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(path_train,
                                                 target_size = (28, 28),
                                                 color_mode = "grayscale",
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(path_test,
                                            target_size = (28, 28),
                                            color_mode = "grayscale",
                                            batch_size = 5,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 339523,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 360)


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)