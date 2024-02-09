import cv2 as cv
import numpy as np

image = cv.imread('basics\media\demo.jpg')
cv.imshow('Image' , image)

# image transformation
def transform(img , x , y):
    mat = np.float32([[1,0,x] , [0,1,y]])
    dimensions = (img.shape[1] , img.shape[0])
    return cv.warpAffine(img , mat , dimensions)

# image Rotation
def rotate(image , rotation_angle , rotpoint = None):
    (height , width) = image.shape[:2]
    if rotpoint == None:
        rotpoint = (width//2 , height//2)
    mat = cv.getRotationMatrix2D(rotpoint , rotation_angle , 1.0)
    dimension = (width , height)
    return cv.warpAffine(image , mat , dimension)

# resizing the image
resized = cv.resize(image , (500 , 500) , interpolation=cv.INTER_AREA)

# transformation
transformed = transform(resized , -100 , -100)
cv.imshow('Transformed' , transformed)

# image rotation
rotated = rotate(resized , 45)
cv.imshow('Rotated' , rotated)

# flipped
flipped = cv.flip(resized , 1)
cv.imshow('Flipped' , flipped)

# cropped
cropped = resized[200:300 , 400:500]
cv.imshow('Cropped' , cropped)

cv.waitKey(0)