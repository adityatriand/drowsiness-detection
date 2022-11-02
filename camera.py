import cv2 as cv
import numpy as np
from retinex import Retinex
from tensorflow.keras.models import load_model

faceDetect = cv.CascadeClassifier('support/haarcascade_frontalface_default.xml')

class Video(object):

    def __init__(self):
        self.video = cv.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def img_estim(self, img, thrshld=130):
        is_dark = np.mean(img) < thrshld
        return True if is_dark else False

    def drowsiness_face(self, image):
        IMG_SIZE = 145
        is_image_dark = self.img_estim(image)
        img_enh = image
        if is_image_dark:
            img_enh = Retinex(image).msrcr(image)
        fix_img = cv.resize(img_enh, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
        fix_img = fix_img / 255
        fix_img = fix_img.reshape(-1, 145, 145, 3)
        model = load_model('support/model.h5')
        pred = model.predict(fix_img)
        label_map = ['drowsy', 'notdrowsy']
        pred = np.argmax(pred)
        final_pred = label_map[pred]
        return final_pred

    def get_frame(self):
        ret,frame = self.video.read()
        a = 30
        predict = -1
        faces = faceDetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            if len(faces) != 0:
                (x, y, w, h) = faces[0]
                cropped = frame[y:y + h+a, x:x + w+a]
                predict = self.drowsiness_face(cropped)
        ret,jpg = cv.imencode('.jpg',frame)
        return jpg.tobytes(), predict