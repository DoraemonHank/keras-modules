import numpy as np
import h5py
import math
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

from PIL import Image
from PIL import ImageEnhance
from os import walk

import cv2

from sklearn.utils import shuffle

def opencv_to_PIL(op_image):
    PIL_image = Image.fromarray(cv2.cvtColor(op_image,cv2.COLOR_BGR2RGB)) 
    return PIL_image

def PIL_to_opencv(PIL_image):
    op_image = cv2.cvtColor(np.asarray(PIL_image),cv2.COLOR_RGB2BGR)
    return op_image
    

def random_scale(image):
    image = PIL_to_opencv(image)
    h,w,c = image.shape
    scale = int(13*np.random.rand())
    image = cv2.resize(image[scale:w-scale,scale:h-scale], (w, h))
# 	image[0:100,100:200] = cv2.resize(image[scale:100-scale,100+scale:200-scale], (100, 100))
    return opencv_to_PIL(image)

def random_rotate(image, angle_range):
    image = PIL_to_opencv(image)
    h,w,c = image.shape
    angle = angle_range * 2 * (np.random.rand() - 0.5)
    rot_mat = cv2.getRotationMatrix2D((50, 50), angle, 1.0)
    image[0:h,0:w] = cv2.warpAffine(image[0:h,0:w], rot_mat, (w, h))
    # image[0:100,100:200] = cv2.warpAffine(image[0:100,100:200], rot_mat, (100, 100))
    return opencv_to_PIL(image)

def random_translate(image, range_x, range_y,range_f):
    image = PIL_to_opencv(image)
    h,w,c = image.shape
    trans_f = range_f * 2 * (np.random.rand() - 0.5)
    trans_x = range_x * 2 * (np.random.rand() - 0.5)
    trans_y = range_y * 2 * (np.random.rand() - 0.5)
    trans_m = np.float32([[1, 0, 0], [0, 1, trans_y]])
    trans_mf = np.float32([[1, 0, trans_f+trans_x], [0, 1, 0]])
    image[:,0:w] = cv2.warpAffine(image[:,0:w], trans_mf, (w, h))
    trans_mf = np.float32([[1, 0, -trans_f+trans_x], [0, 1, 0]])
    # image[:,100:200] = cv2.warpAffine(image[:,100:200], trans_mf, (100, 100))
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return opencv_to_PIL(image)

def random_brightness(image):
    image = PIL_to_opencv(image)
    h,w,c = image.shape
    base = np.random.rand()*0.4 + 0.8
    image = np.int32(image) * base
    image = np.clip(image, 0, 255)
    return opencv_to_PIL(np.uint8(image))


# image = cv2.imread('./pet_data/training_set/dogs/dog.1001.jpg')
# image = Image.open('./pet_data/training_set/dogs/dog.1001.jpg')
# image = random_scale(image)
# image = random_rotate(image,10)
# image = random_translate(image, 10,15,5)
# image = random_brightness(image)
# image = image.resize((64, 64),Image.ANTIALIAS)   
# # cv2.imshow('sdfsd',random_scale(image))
# plt.imshow(image)



def create_dataset(path,set_name,width, height,classes):
    
    image_array_buffer = []
    mypath = path + '/' + set_name # './pet_data/training_set'

    label = 0
    label_buffer = []
    for root, dirs, files in walk(mypath):
        if(len(files)):
            for i in range(len(files)):
                image = Image.open(root + '/' + files[i])
                if('train' in set_name):
                    img_tmp = image
                print(root + '/' + files[i])
                
                if('train' in set_name):
                    print('auto')
                    image = random_scale(image)
                    image = random_rotate(image,10)
                    image = random_translate(image, 10,15,5)
                    image = random_brightness(image)

                image = image.resize((width, height),Image.ANTIALIAS)                                
                image_array = np.array(image)
                image_array_buffer.append(image_array)
                label_buffer.append(label)
                
                if('train' in set_name):
                    img_tmp = img_tmp.resize((width, height),Image.ANTIALIAS)
                    image_array = np.array(img_tmp)
                    image_array_buffer.append(image_array)
                    label_buffer.append(label)

            label = label + 1

    f = h5py.File(set_name + '.h5',"w")
    f["list_classes"] = np.arange(classes)
    
    
    if('train' in set_name):
        set_xy = 'train'
    else:
        set_xy = 'test'
    f[set_xy + "_set_x"] = image_array_buffer
    f[set_xy + "_set_y"] = label_buffer
    for key in f.keys():
        print(key)
        print(f[key].name)
        print(f[key].shape)
        print(f[key].value)
    f.close()
   
    
# create_dataset('./pet_data','training_set',2)
# create_dataset('./pet_data','test_set',2)
    

# train_dataset = h5py.File('./pet_data' + '/' + 'training_set.h5', "r")
# train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
# train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

# train_set_x_orig, train_set_y_orig = shuffle(train_set_x_orig, train_set_y_orig, random_state=0)

# plt.imshow(train_set_x_orig[500][:])
# print(train_set_y_orig[500])


def load_dataset(path,train_set_name,test_set_name):
    train_dataset = h5py.File(path + '/' + train_set_name, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels    
    train_set_x_orig, train_set_y_orig = shuffle(train_set_x_orig, train_set_y_orig, random_state=0)
    
    test_dataset = h5py.File(path + '/' + test_set_name, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels    
    test_set_x_orig, test_set_y_orig = shuffle(test_set_x_orig, test_set_y_orig, random_state=0)

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig,classes):

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, classes).T
    Y_test = convert_to_one_hot(Y_test_orig, classes).T

    return X_train, Y_train, X_test, Y_test

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def plot_model_history(model_history,acc_name):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(model_history.history[acc_name])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc_name]) + 1), len(model_history.history[acc_name]) / 10)
    axs[0].legend(['train'], loc='best')

    axs[1].plot(model_history.history['loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train'], loc='best')
    plt.savefig('./images/acc_loss.png')

    print('Plots saved!')