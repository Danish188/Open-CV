import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('basics/demo.jpg')
img = cv.resize(img, (500, 600), interpolation = cv.INTER_AREA)
cv.imshow('Image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale' , gray)

circle = cv.circle(blank , (img.shape[1]//2 , img.shape[0]//2),
                 150 , 255 , -1)

mask = cv.bitwise_and(img , img , mask = circle)
cv.imshow("Mask", mask)

# Grayscale Histogram
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256])

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

plt.figure()
plt.title('GBR Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img] , [i] , circle , [256] , [0,256])
    plt.plot(hist , color = col)
    plt.xlim([0, 256])
plt.show()

cv.waitKey(0)