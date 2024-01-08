import os
import cv2 as cv
import numpy as np

p = []

for i in os.listdir(r'D:\OpenCV\projects\face_recognition\train'):
    p.append(i)

print(p)

DIR = r'D:\OpenCV\projects\face_recognition\train'

har_cascade = cv.CascadeClassifier(r'projects\face_detection\haar_face.xml')

features = []
labels = []

def create_train():
    for person in p:
        path = os.path.join(DIR , person)
        label = p.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path , img)
            
            img_read = cv.imread(img_path)
            gray = cv.cvtColor(img_read , cv.COLOR_BGR2GRAY)
            
            faces_rect = har_cascade.detectMultiScale(gray ,scaleFactor=1.1, minNeighbors=4)
            
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h , x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print('Extracting features and labels done! -----------')

features = np.array(features, dtype='object')
labels = np.array(labels )

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# training the recognizer
face_recognizer.train(features , labels)

face_recognizer.save(r'projects\face_recognition\face_trained.yml')

np.save(r'projects\face_recognition\features.npy' , features)
np.save(r'projects\face_recognition\labels.npy' , labels)