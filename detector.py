import cv2
import numpy as np
import os
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('D:/python-opencv/NORMAL/haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('D:/python-opencv/NORMAL/recognizer/traindataData.yml')
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    
        
        ret,frame = cap.read()
        if ret is True:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
                id,conf=rec.predict(gray[y:y+h,x:x+w])
                print(conf)
                if(id >=1 and id<=20):
                       id = "prashant"
                cv2.putText(frame,text=str(id),org=(x,y),fontFace=font,fontScale=1,color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)    
            cv2.imshow('frame',frame)
            if cv2.waitKey(100) & 0xFF == 27:
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
