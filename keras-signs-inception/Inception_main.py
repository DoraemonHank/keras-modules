from Inception import InceptionV1

from Inception_utils import load_dataset, preprocess_data,plot_model_history
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#Split train and test data
X_train, Y_train, X_test, Y_test = preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

#Data augumentation with Keras tools
# from keras.preprocessing.image import ImageDataGenerator
# img_gen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
#     )



InceptionV1_model = InceptionV1(64,64,3,classes = len(classes))
InceptionV1_model.summary()

InceptionV1_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])
History = InceptionV1_model.fit(X_train, Y_train, epochs = 20, batch_size = 16)


# History = InceptionV1_model.fit_generator(img_gen.flow(X_train*255, y_train, batch_size = 16),steps_per_epoch = len(X_train)/16, validation_data = (X_test,y_test), epochs = 30 )

# plot accuracy/loss
plot_model_history(History,'accuracy')

# evaluate model
preds = InceptionV1_model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))