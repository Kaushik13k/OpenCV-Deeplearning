# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a "temporary script file.
"""

import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining the functions that will do the detection!
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # x, y --> upper left cordinate of pic, w -> width, h -> height
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0,), 2) # cordinate of lower right corner -> (x+w, y+h)
        # region of interst
        roi_gray = gray[y:y+h, x:x+w] #gray_image
        roi_color = frame[y:y+h, x:x+w] #coloured_image
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # x, y --> upper left cordinate of pic, w -> width, h -> height
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # cordinate of lower right corner -> (x+w, y+h)
    return frame

# Doing some face recognition with web cam
video_capture = cv2.VideoCapture(0) # 0--> internal webcam
while True:
    _, frame = video_capture.read()# get last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
