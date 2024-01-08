import cv2 as cv 
import numpy as np

img = cv.imread('basics/demo.jpg')
img = cv.resize(img , (500,600) , interpolation = cv.INTER_AREA)
cv.imshow('Image' , img)

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale' , gray)

# laplacian
lap = cv.Laplacian(gray , cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian' , lap)

# sabel
sob = cv.Sobel(gray , cv.CV_64F , 1 , 0)
soby = cv.Sobel(gray , cv.CV_64F , 0 , 1)
combined = cv.bitwise_or(sob , soby)

cv.imshow('Sobel X' , sob)
cv.imshow('Sobel Y' , soby)
cv.imshow('Combined Sobel' , combined)

canny = cv.Canny(gray , 150 , 175)
cv.imshow('Canny' , canny)

cv.waitKey(0)