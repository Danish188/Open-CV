import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('basics/media/demo.jpg')
image = cv.resize(image , (500 , 600) , interpolation=cv.INTER_AREA)
cv.imshow('Image' , image)

# plt.imshow(image)
# plt.show()

# BGR to HSV
hsv = cv.cvtColor(image , cv.COLOR_BGR2HSV)
cv.imshow('HSV' , hsv)

# BGR to L+a+b 
lab = cv.cvtColor(image , cv.COLOR_BGR2LAB)
cv.imshow('LAB' , lab)

# BGR to RGB
rgb = cv.cvtColor(image , cv.COLOR_BGR2RGB)
cv.imshow('RGB' , rgb)

# hsv to BGR
hsv_bgr = cv.cvtColor(hsv , cv.COLOR_HSV2BGR)
cv.imshow('HSV --> BGR' , hsv_bgr)

# hsv to BGR
lab_bgr = cv.cvtColor(lab , cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR' , lab_bgr)

cv.waitKey(0)