import cv2 as cv 
import numpy as np

img = cv.imread('basics/media/demo.jpg')
img = cv.resize(img , (500 , 600) , interpolation=cv.INTER_AREA)
cv.imshow('Image' , img)

# Average Blur
avg = cv.blur(img , (3,3))
cv.imshow('Average Blur' , avg)

# Gaussian Blur
gauss = cv.GaussianBlur(img , (3,3) , 0)
cv.imshow('Gaussian Blur' , gauss)

# Median Blur
med = cv.medianBlur(img , 3)
cv.imshow('Median Blur' , med)

# Bilateral Blur
bilateral = cv.bilateralFilter(img , 10 , 35 , 25)
cv.imshow('Bilateral Blur' , bilateral)

cv.waitKey(0) 