import cv2 as cv

def reco_webcam():
    """s
    _ brief: this functions starts the webcam and detects eyes and faces in the video
    _ param: none
    _ return: the video from the webcam with blocks of color around faces and eyes
    """
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    video_capture = cv.VideoCapture(0)
    while True :
        ret, frame = video_capture.read()
        frame = cv.flip(frame, 1)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv.imshow('img',frame)
        if cv.waitKey(1) == 27:
            break  # esc to quit
    cv.waitKey(0)
    cv.destroyAllWindows()






