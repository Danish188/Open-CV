import cv2 as cv
import numpy as np
from PIL import Image 

img = cv.imread('basics\media\demo.jpg')
img = cv.resize(img, (500,600), interpolation = cv.INTER_AREA)

blank = np.zeros((500,500,3) , dtype='uint8')
cv.imshow('Blank', blank)

cv.rectangle(blank , (0,0), (200, 500) , (0,255,0), thickness=-1)
cv.imshow('Rectangle', blank)

cv.line(blank , (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,0,0), thickness=1)
cv.imshow('line', blank)

cv.putText(blank , 'Hello World' , (0,250) , cv.FONT_HERSHEY_TRIPLEX, 1 , (255,255,255), thickness=3)
cv.imshow('Text', blank)

cv.waitKey(0)