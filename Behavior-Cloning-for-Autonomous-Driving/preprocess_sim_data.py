# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:08:09 2019
@author: elif.ayvali
removes a percentage of straight steeering data
augments the remaining data set with left and right camera images
puts all data in a pickle file
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import math
import os 

save_data=True


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def process_data(img_path , log_file, plotting=False):  
    
    #processing params
    correction = 0.35
    w_size=3
    drop_ratio=0.8

    camera_images=[]
    steering_commands=[]
    dataframe = pd.read_csv(log_file,index_col=False)
    imgs_center = dataframe["center"].iloc[1:,].values
    imgs_left = dataframe["left"].iloc[1:,].values
    imgs_right = dataframe["right"].iloc[1:,].values
    steering = dataframe["steering"].iloc[1:,].values
    steering_smooth =moving_average(steering,w_size)
    
    
    for img_center, img_left, img_right, steering_command in zip(imgs_center, imgs_left, imgs_right, steering_smooth):
        _, center_imgfile = os.path.split(img_center)
        _, left_imgfile = os.path.split(img_left)
        _, right_imgfile = os.path.split(img_right)
    
        # drop %80 of the  datapoints for steering straight
        if (math.isclose(steering_command, 0, abs_tol=0.05) and np.random.random()<drop_ratio):
            continue          
#    
        # center image
        camera_images.append(center_imgfile)
        steering_commands.append(steering_command)
       
        # left image      
        camera_images.append(left_imgfile)
        steering_commands.append(steering_command+correction) 
        
        # right image
        camera_images.append(right_imgfile)
        steering_commands.append(steering_command-correction) 
    if plotting is True:
        plt.figure()        
        plt.plot(steering[2000:2200],color='k')
        plt.plot(steering_smooth[2000:2200],color='r')   
        plt.figure()
        plt.hist(steering)
        plt.figure()
        plt.hist(steering_commands)  
             
    return camera_images, steering_commands     

log_file = './sim_data/driving_log.csv'
img_path= './sim_data/IMG/'
    

camera_images,steering_commands=process_data(img_path,log_file, plotting=True)
print('...Preprocessed data')

#save_data
if save_data is True:
    sim_data=dict()
    sim_data['images'] = camera_images
    sim_data['steering'] = steering_commands
    with open('./sim_data/sim_data.p', mode='wb') as f:  
    	pickle.dump(sim_data, f)
    print('...Saved data')
