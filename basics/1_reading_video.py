import cv2 as cv

capture = cv.VideoCapture('basics\media\car.mp4')

def rescale(f , scale = 0.50):
    width = int(f.shape[1] * scale)
    height = int(f.shape[0] * scale)
    dimension = (width , height)
    return cv.resize(f , dimension , interpolation=cv.INTER_AREA)

while True:
    istrue , frame = capture.read()
    frame_resized = rescale(frame)
    cv.imshow('Video' ,frame)
    cv.imshow('rescaled_video' , frame_resized)
    if cv.waitKey(20) & 0xff == ord('z'):
        break
    
capture.release()
cv.destroyAllWindows()