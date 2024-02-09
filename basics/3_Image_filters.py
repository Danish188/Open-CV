import cv2 as cv
import numpy as np 

def rescale(f , scale = 0.50):
    width = int(f.shape[1] * scale)
    height = int(f.shape[0] * scale)
    dimension = (width , height)
    return cv.resize(f , dimension , interpolation=cv.INTER_AREA)

#video
video = cv.VideoCapture('basics\media\car.mp4')

#image
# image = cv.imread('basics\media\demo.jpg')
# image = rescale(image , 0.18)
# cv.imshow('Image' , image)

while True:
    index, frame = video.read()
    rescaled = rescale(frame)
    grayscale = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grayscale, (7,7), cv.BORDER_DEFAULT)
    edges = cv.Canny(blur, 100, 175)
    dilate = cv.dilate(edges, (2,2), iterations = 3)
    cv.imshow('Video Frame', dilate)
    if cv.waitKey(20) & 0xFF == ord('y'):
        break
    
video.release()
cv.destroyAllWindows()

# GrayScale
# grayscale = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale' , grayscale)

# # blurring the image
# blur = cv.GaussianBlur(image , (3,3) , cv.BORDER_DEFAULT)
# cv.imshow('Blur' , blur)

# # edges
# edges = cv.Canny(blur , 125 , 175)
# cv.imshow('Edges' , edges)

# # dillation
# dilation = cv.dilate(edges , (7,7) , iterations = 3)
# cv.imshow('Dilated Image' , dilation)

# cv.waitKey(0)