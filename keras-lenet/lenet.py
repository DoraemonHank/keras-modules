import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,AveragePooling2D,concatenate,Input, concatenate
from keras.models import Model,load_model
from keras.optimizers import Adam


  
#Build Lenet model
def Lenet(width, height, depth, classes):
    
    inpt = Input(shape=(width,height,depth))
    x = Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='tanh')(inpt)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(120, activation='tanh')(x)
    x = Dense(84, activation='tanh')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input=inpt,output=x)
    
    return model

def Lenet2(width, height, depth, classes):
    
    inpt = Input(shape=(width,height,depth))
    x = Conv2D(32, (3, 3), activation = 'relu')(inpt)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input=inpt,output=x)
    
    return model