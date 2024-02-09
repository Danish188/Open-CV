import cv2 as cv
import numpy as np 

image = cv.imread('basics/media/demo.jpg')
image = cv.resize(image , (500 , 600) , interpolation=cv.INTER_AREA)
cv.imshow('Image' , image)

# blank image
blank = np.zeros(image.shape, dtype='uint8')
cv.imshow('Blank' , blank)

# graysclae
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
cv.imshow('Gray' , gray)

# blur image
blur = cv.GaussianBlur(gray , (5,5) , cv.BORDER_DEFAULT)
cv.imshow('Blur' , blur)

# canny edge detection
# canny = cv.Canny(blur , 125 , 175)
# cv.imshow('canny' , canny)

# thresholding
ret , thresh = cv.threshold(blur , 125 , 255 , cv.THRESH_BINARY)
cv.imshow("Thresh" , thresh)

# countours in image
countours , hierarchies = cv.findContours(thresh , cv.RETR_LIST , cv.CHAIN_APPROX_SIMPLE)
print(f'{len(countours)} countours detected')

# drawing countours on blank image
cv.drawContours(blank , countours , -1 , (0,255,0) , 1)
cv.imshow('Blank' , blank)

cv.waitKey(0)