import cv2 as cv
import numpy as np

img = cv.imread('basics/media/demo.jpg')
img = cv.resize(img , (500,600), interpolation = cv.INTER_AREA)
cv.imshow("Image" , img)

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('Gray' , gray)

# simple thresholding
threshold, thresh = cv.threshold(gray , 100 , 255 , cv.THRESH_BINARY)
cv.imshow('Thresholded' , thresh)

# inverse thresholding
threshold, thresh_inv = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
cv.imshow('Inverse Thresholded' , thresh_inv)

# Adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255,
                                       cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, 11, 
                                       3)
cv.imshow('Adaptive Thresholding' , adaptive_thresh)

cv.waitKey(0)