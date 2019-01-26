# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:09:22 2019

@author: Aaditya
"""

import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('C:\ProgramData\Anaconda3\Lib\site-packages\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels= pickle.load(f)
    labels= {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id, conf = recognizer.predict(roi_gray)
        if conf>=45: #and conf <=85:
            #print(id)
            # print(labels[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            
            
        img_item = "1.jpg"
        cv2.imwrite(img_item, roi_color)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==13:
        break
        
cap.release()
cv2.destroyAllWindows()