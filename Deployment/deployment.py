import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_righteye_2splits.xml')
model = load_model(r'C:\Users\yousef walid\OneDrive\Desktop\project samsung\final project\final\cnnmodel.h5')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
import streamlit as st
from PIL import Image
with st.sidebar:
   st.header("Eye-Detection")
   image = Image.open("detection.jpg")
   st.image(image)
   st.subheader("this model to predict open or close eye you can use it by upload your image ,and it's respond with open or closed")
   st.header(" BY/ENG.youssef and ENG.abdelziz")
   genre = st.radio(
    "predicted or live ??",
    ('predicted', 'live'))
if  genre=='live' :
        video_file = open('finalcv.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
else :
        from keras.preprocessing.image import load_img
        import streamlit as st
        from tempfile import NamedTemporaryFile
        st.subheader("please upload rgb an image :)")
        buffer = st.file_uploader("")
        temp_file = NamedTemporaryFile(delete=False)
        if buffer:
               
                image = Image.open(buffer)
                st.image(image)
                temp_file.write(buffer.getvalue())
                frame=load_img(temp_file.name)
                gray = cv2.cvtColor(np.array(frame, dtype='uint8'), cv2.COLOR_RGB2GRAY)
                faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
                left_eye = leye.detectMultiScale(gray)
                right_eye =  reye.detectMultiScale(gray)
                frame=np.float32(frame)
                cv2.rectangle(frame, (0,1280-50) , (200,1280) , (0,0,0) , thickness=cv2.FILLED )
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            
                for (x,y,w,h) in right_eye:
                    r_eye=frame[y:y+h,x:x+w]
              
                    r_eye = cv2.cvtColor(np.float32(r_eye),cv2.COLOR_BGR2GRAY)
                    r_eye = cv2.resize(r_eye,(24,24))
                    r_eye= r_eye/255
                    r_eye=  r_eye.reshape(24,24,-1)
                    r_eye = np.expand_dims(r_eye,axis=0)
                    predict_x=model.predict(r_eye) 
                    rpred=np.argmax(predict_x,axis=1)
                    if(rpred[0]==1):
                        lbl='Alert' 
                    if(rpred[0]==0):
                        lbl='Sleepy!'
                    break
            
                for (x,y,w,h) in left_eye:
                    l_eye=frame[y:y+h,x:x+w]
                    l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                    l_eye = cv2.resize(l_eye,(24,24))
                    l_eye= l_eye/255
                    l_eye=l_eye.reshape(24,24,-1)
                    l_eye = np.expand_dims(l_eye,axis=0)
                    predict_x=model.predict(l_eye) 
                    lpred=np.argmax(predict_x,axis=1)
                    if(lpred[0]==1):
                        lbl='Alert'   
                    if(lpred[0]==0):
                        lbl='Sleepy!'
                    break
            
                if(rpred[0]==0 and lpred[0]==0):
                    
                    st.subheader("the result for a model is : closed")
              
                else:
                    st.subheader("the result for a model is : open")
               
        