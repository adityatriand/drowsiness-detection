import cv2 as cv
import numpy as np
from retinex import Retinex
import dlib
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

    def extract_feat(self,image):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('support/shape_predictor_68_face_landmarks.dat')
        shape = 0
        # Detect the face
        rects = detector(image, 1)
        # Detect landmarks for each face
        for rect in rects:
            # Get the landmark points
            shape = predictor(image, rect)
            # Convert it to the NumPy Array
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)
            shape = shape_np

            # Display the landmarks
            for i, (x, y) in enumerate(shape):
                # Draw the circle to mark the keypoint
                cv.circle(image, (x, y), 1, (0, 0, 255), -1)
        return image

    def drowsiness_face(self, image):
        IMG_SIZE = 145
        is_image_dark = self.img_estim(image)
        img_enh = image
        if is_image_dark:
            img_enh = Retinex().msrcr(image)
        img_enh = self.extract_feat(img_enh)
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
        predict = 'no face'
        faces = faceDetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            if len(faces) != 0:
                (x, y, w, h) = faces[0]
                cropped = frame[y:y + h+a, x:x + w+a]
                predict = self.drowsiness_face(cropped)
            cv.putText(frame, predict, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        ret,jpg = cv.imencode('.jpg',frame)
        return jpg.tobytes()