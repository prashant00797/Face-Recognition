import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier('D:/python-opencv/NORMAL/haarcascade_frontalface_default.xml')

path = 'D:/python-opencv/NORMAL/data'

def getImagesWithID(path):
    imagePath = [os.path.join(path,f) for f in os.listdir(path)]

    faceSamples = []
    ids = []

    for imagePath in imagePath:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img)

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(Id)
            print(Id)
            cv2.imshow('train',img_numpy)
            cv2.waitKey(10)
    return faceSamples,ids

faces,ids = getImagesWithID(path)

recognizer.train(faces, np.array(ids))

recognizer.save('D:/python-opencv/NORMAL/recognizer/traindataData.yml')

cv2.destroyAllWindows()
