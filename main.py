import cv2 as cv

video = cv.VideoCapture(0)

faceDetect = cv.CascadeClassifier('support/haarcascade_frontalface_default.xml')

while True:
    ret,frame = video.read()
    faces = faceDetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255),2)
    cv.imshow("Frame", frame)
    k = cv.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv.destroyAllWindows()