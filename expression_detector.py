# -*- coding: utf-8 -*-
"""
Created on Thu Jul  12 16:13:17 2021

@author: 13kau
"""
import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # x, y --> upper left cordinate of pic, w -> width, h -> height
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0,), 2) # cordinate of lower right corner -> (x+w, y+h)
        # region of interst
        roi_gray = gray[y:y+h, x:x+w] #gray_image
        roi_color = frame[y:y+h, x:x+w] #coloured_image
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color,(sx, sy),(sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

video_capture = cv2.VideoCapture(0) # 0--> internal webcam
while True:
    _, frame = video_capture.read()# get last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

