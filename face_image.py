import cv2
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
image = cv2.imread('test.png')
gray_scaling = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(gray_scaling, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+h, y+w), (0,255,0), 2)
    cv2.imshow('image', image) cv2.waitKey()