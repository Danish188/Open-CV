import cv2 as cv    
import numpy as np

img = cv.imread('basics/media/demo.jpg')
img = cv.resize(img , (500 , 600) , interpolation = cv.INTER_AREA)
cv.imshow('IMAGE ' , img)

blank = np.zeros(img.shape[:2] , dtype='uint8')
mask = cv.rectangle(blank ,
                    (img.shape[1]//2 , img.shape[0]//2),
                    (img.shape[1]//2 + 100 , img.shape[0]//2 + 100),
                    255 , -1)
cv.imshow('Mask' , mask)

masked = cv.bitwise_not(img , img , mask = mask)
cv.imshow('Masked Image' , masked)

cv.waitKey(0)