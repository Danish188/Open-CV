import cv2 as cv

img = cv.imread('projects/face_detection/demo1.jpeg')
img = cv.resize(img , (500, 600), interpolation=cv.INTER_AREA)
cv.imshow('Image' , img)

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('GrayScale' , gray)

har_cascade = cv.CascadeClassifier('projects/face_detection/haar_face.xml')
face_rect = har_cascade.detectMultiScale(gray ,
                                         scaleFactor=1.1 ,
                                         minNeighbors=3)

print(f'Numbers of detected faces {len(face_rect)}')

for (x,y,w,h) in face_rect:
    cv.rectangle(img , (x,y) , (x+w,y+h) , (0,255,0) , 2)

cv.imshow('Detected face' , img)

cv.waitKey(0)