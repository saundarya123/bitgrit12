# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:08:08 2019

@author: Aaditya
"""

import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('C:\ProgramData\Anaconda3\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids = {}
y_labels = []
x_train = []



for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" "," ").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id = label_ids[label]
            #print(label_ids)
                         
            #y_labels.append(label)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            size= (550,550)
            final_image=pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array,1.5,5)
            
            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)
                
#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")