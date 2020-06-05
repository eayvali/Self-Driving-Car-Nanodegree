# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:21:51 2019

@author: elif.ayvali
"""

#import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense,  Lambda, Cropping2D, SpatialDropout2D
from keras.utils import plot_model
from keras.callbacks import History, ModelCheckpoint
from keras.optimizers import Adam
import cv2
from math import ceil
import pickle
import time


img_folder='./sim_data/IMG/'
log_pickle_file='./sim_data/sim_data.p'

def train_batch_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            cam_images = []
            steering_commands = []
            for batch_sample in batch_samples:
                img_name = img_folder+batch_sample[0]
                steering = float(batch_sample[1])  
                image = plt.imread(img_name)#RGB
                #add car iamge, steering
                cam_images.append(image)
                steering_commands.append(steering)
                #augment by flipping the image
                cam_images.append(cv2.flip(image,1))
                steering_commands.append(-steering)
            X_batch = np.array(cam_images)
            y_batch = np.array(steering_commands)
            yield sklearn.utils.shuffle(X_batch, y_batch)
            
def valid_batch_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            cam_images = []
            steering_commands = []
            for batch_sample in batch_samples:
                img_name = img_folder+batch_sample[0]
                steering = float(batch_sample[1])  
                image = plt.imread(img_name)#RGB
                #add car iamge, steering
                cam_images.append(image)
                steering_commands.append(steering)
            X_batch = np.array(cam_images)
            y_batch = np.array(steering_commands)
            yield sklearn.utils.shuffle(X_batch, y_batch)

def TestNetwork(input_shape):
    #used for debugging#
    model = Sequential()
    #normalization
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((60,20), (0,0))))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def commaai_model(input_shape):
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))
  model.add(Cropping2D(cropping=((60, 20), (0, 0))))
  model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding="same", name='Conv1'))
  model.add(ELU())
  model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same", name='Conv2'))
  model.add(ELU())
  model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same", name='Conv3'))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))
  return model


def NvidiaNet(input_shape):
  
    def resize_normalize(image):
        from keras.backend import tf as ktf    
        """
        resizes to 64*64 px and scales pixel values to [0, 1].
        """
        resized = ktf.image.resize_images(image, (64, 64))
        #normalize 0-1
        resized = resized/255.0 - 0.5

        return resized
    model = Sequential()
    # normalization
    #model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    #resize image        
    model.add(Lambda(resize_normalize, input_shape=(80, 320, 3), output_shape=(64, 64, 3)))
    # Convolutional layers
    model.add(Convolution2D(24, (5, 5), strides=(2, 2),padding="same", activation="relu", name='Conv1'))
    model.add(SpatialDropout2D(.2))     
    model.add(Convolution2D(36, (5, 5), strides=(2, 2),padding="same", activation="relu", name='Conv2'))
    model.add(SpatialDropout2D(.2))  
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="same", activation="relu", name='Conv3'))    
    model.add(SpatialDropout2D(.2))     
    model.add(Convolution2D(64, (3, 3), padding="same",  activation="relu", name='Conv4'))    
    model.add(SpatialDropout2D(.2))     
    model.add(Convolution2D(64, (3, 3), padding="same", activation="relu", name='Conv5'))    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model

def train_car_sim():
    samples = []
    # Load driving data
    with open(log_pickle_file, mode='rb') as f:
        sim_data = pickle.load(f)
        
    samples=list(zip(sim_data['images'],sim_data['steering']))
    #debugging on CPU
#    samples=samples[:1000]
    
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    # Set the batch size
    batch_size=32
    # compile and train the model using the generator function
    train_generator = train_batch_generator(train_samples, batch_size=batch_size)
    validation_generator = valid_batch_generator(validation_samples, batch_size=batch_size)
    
    input_shape = (160, 320, 3)  # Original image size
    
    # model = TestNetwork(img_size)
    model=NvidiaNet(input_shape)
    print(model.summary())
    # Visualize model and save it to disk
    plot_model(model, to_file='./sim_data/model.png', show_shapes=True, show_layer_names=False)
    lr=1e-04
    
    model.compile(loss='mse', optimizer=Adam(lr), metrics=['accuracy'])
    
    h = History()
    # Save best model weight: need to add checkpoint to vallbacks
#    checkpoint = ModelCheckpoint('./sim_data/best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

    train_start_time = time.time()

    model.fit_generator(train_generator, 
                steps_per_epoch=ceil(len(train_samples)/batch_size), 
                validation_data=validation_generator, 
                validation_steps=ceil(len(validation_samples)/batch_size), 
                epochs=20, verbose=1, callbacks=[h])

    elapsed_time = time.time() - train_start_time
    print('Training time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))    
 
    
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
#     plt.show()
    plt.savefig('./sim_data/model_loss.png')
    
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
#    plt.show()
    plt.savefig('./sim_data/model_accuracy.png')

    model.save('model.h5')    
    
    # Save model architecture to model.json, model weights to model.h5
    json_string = model.to_json()
    with open('./sim_data/model.json', 'w') as f:
    	f.write(json_string)
    model.save_weights('./sim_data/model.h5')
    # Save training history
    with open('./sim_data/train_hist.p', 'wb') as f:
    	pickle.dump(h.history, f)
        
             

if __name__ == '__main__':
	print('..Training autonomous car simulator')
	train_car_sim()
	print('..Completed training')