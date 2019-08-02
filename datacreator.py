import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('D:/python-opencv/NORMAl/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

id = input('Enter the user id')
sampleNum = 0

while True:
        
        ret,frame = cap.read()
        if ret is True:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
                cv2.imwrite('D:/python-opencv/NORMAL/data/User' + str(id) + '.' + str(sampleNum) + '.jpg',gray[y:y+h,x:x+w])
                sampleNum = sampleNum + 1
                cv2.imshow('frame',frame)
            if cv2.waitKey(100) & 0xFF == 27:
                break
            elif sampleNum >= 21:
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
