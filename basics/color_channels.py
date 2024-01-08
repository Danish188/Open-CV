import cv2 as cv
import numpy as np

img = cv.imread('basics/demo.jpg')
img = cv.resize(img , (500 , 600) , interpolation=cv.INTER_AREA)
cv.imshow('DEMO' , img)

blank = np.zeros(img.shape[:2] , dtype = 'uint8')

b , g, r = cv.split(img)

blue = cv.merge([b, blank , blank])
green = cv.merge([blank , g , blank])
red = cv.merge([blank , blank , r])

cv.imshow('Blue' , blue)
cv.imshow('Green' , green)
cv.imshow('Red' , red)

print(img.shape)
print(g.shape)
print(b.shape)
print(r.shape)

merge = cv.merge([b,g,r])
cv.imshow('Merged image' , merge)


cv.waitKey(0)