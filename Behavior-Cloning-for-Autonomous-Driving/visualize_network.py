# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:50:35 2019

@author: elif.ayvali
"""

from keras.models import  Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Model reconstruction from JSON file
with open('./final 05_30/model.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('./final 05_30/model.h5')


layer1 = Model(inputs=model.input, outputs=model.get_layer('Conv1').output)
layer2 = Model(inputs=model.input, outputs=model.get_layer('Conv2').output)

img_name = './data/IMG/center_2016_12_01_13_38_51_331.jpg'
img=plt.imread(img_name)
img = np.expand_dims(img, axis=0)
steering = model.predict(img)
visual_layer1, visual_layer2 = layer1.predict(img), layer2.predict(img)



plt.figure()
plt.plot()
plt.imshow(img.squeeze())
plt.title('steering angle : '+ str(np.round(float(steering),3)))
plt.axis('off')
plt.show()


plt.figure(figsize=(12,8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    temp = visual_layer1[0, :, :, i]
    temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
    plt.imshow(temp)
    plt.axis('off')
plt.show()
plt.savefig('./layer1.png')




plt.figure(figsize=(12,8))
for i in range(32):
    plt.subplot(8, 4, i+1)
    temp = visual_layer2[0, :, :, i]
    temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
    plt.imshow(temp)
    plt.axis('off')
plt.show()
plt.savefig('./data/layer2.png')
