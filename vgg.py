
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import pickle

vgg_path="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
def create_model():
    model = Sequential()
    model.add(tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, pooling='avg'))
    model.add(Dropout(0.50))
    model.add(Dense(10,activation="softmax"))
    return model

def get_fitted_data_generator(data):
    data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                   height_shift_range=0.1, zoom_range=0.1)
    data_generator.fit(data)
    return data_generator

def model_training_generator(model, x_train, y_train, epochs=1, batch=32, validation=False,x_val=None,y_val=None):
    image_count = np.size(x_train,0)
    training_data_generator = get_fitted_data_generator(x_train)

    if validation:
        return model.fit_generator(training_data_generator.flow(x_train,y_train,batch_size=batch),steps_per_epoch=(image_count//batch),epochs=epochs,validation_data=(x_val,y_val),verbose=1)
    else:
        return model.fit_generator(training_data_generator.flow(x_train,y_train,batch_size=batch),steps_per_epoch=(image_count//batch),epochs=epochs,verbose=1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./MNIST_DATA')
x_train = x_train.reshape((x_train.shape[0],28,28,1)) /255.0
x_test = x_test.reshape((x_test.shape[0],28,28,1)) /255.0
x_train = (x_train - 0.5) /0.5
x_test = (x_test - 0.5)/0.5
x_train = tf.image.resize(x_train,(32,32),method=tf.image.ResizeMethod.BILINEAR) 
x_test = tf.image.resize(x_test,(32,32),method=tf.image.ResizeMethod.BILINEAR)
#x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train),name=None)
#x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test),name=None)
x_train = tf.convert_to_tensor(x_train.numpy().repeat(3,-1))
x_test = tf.convert_to_tensor(x_test.numpy().repeat(3,-1))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


model = create_model()
epochs = 15
lrate = 0.001
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model_training_generator(model,x_train,y_train,validation=True,x_val=x_val,y_val=y_val,epochs=epochs)
model_json = model.to_json()
with open("vgg_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(os.getcwd(), 'vgg_model.h5'))
pickle.dump(hist.history,open("hist.p","wb"))
results = model.evaluate(x_test,y_test,batch_size=32)
print("ResNet Evaluation")
print("test loss, test acc: ", results)
