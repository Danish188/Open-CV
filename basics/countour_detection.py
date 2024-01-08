import cv2 as cv
import numpy as np 

image = cv.imread('basics/demo.jpg')
image = cv.resize(image , (500 , 600) , interpolation=cv.INTER_AREA)
cv.imshow('Image' , image)

blank = np.zeros(image.shape, dtype='uint8')
cv.imshow('Blank' , blank)

gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
cv.imshow('Gray' , gray)

blur = cv.GaussianBlur(gray , (5,5) , cv.BORDER_DEFAULT)
cv.imshow('Blur' , blur)

canny = cv.Canny(blur , 125 , 175)
cv.imshow('canny' , canny)

# ret , thresh = cv.threshold(gray , 125 , 255 , cv.THRESH_BINARY)
# cv.imshow("Thresh" , thresh)

countours , hierarchies = cv.findContours(canny , cv.RETR_LIST , cv.CHAIN_APPROX_SIMPLE)
print(f'{len(countours)} countours detected')

cv.drawContours(blank , countours , -1 , (0,255,0) , 1)
cv.imshow('Blank' , blank)

cv.waitKey(0)