import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier(r'projects\face_detection\haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfeld', 'Madonna', 'Mindy Kaling']
# features = np.load(r'projects\face_recognition\features.npy')
# labels = np.load(r'projects\face_recognition\labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'projects\face_recognition\face_trained.yml')

img = cv.imread(r'projects\face_recognition\val\Madonna\httpcdnfuncheapcomwpcontentuploadsVOGUEjpg.jpg')
gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('Gray' , gray)

faces_rect = haar_cascade.detectMultiScale(gray , 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h , x:x+w]
    
    label , confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    
    cv.putText(img , str(people[label]) , (20,20) , cv.FONT_HERSHEY_COMPLEX,
               1.0, (0,255,0) , thickness = 2)
    cv.rectangle(img , (x,y) , (x+w, y+h) , (0,255,0) , 2)
cv.imshow('Detected Image' , img)
cv.waitKey(0)