from flask import Flask, render_template, Response, request
from camera import Video
from retinex import Retinex
import cv2 as cv

class App:

    def __init__(self, name):
        self.app = Flask(name)

        @self.app.route('/')
        def index():
            return self.__index()

        @self.app.route('/guideapp')
        def guideapp():
            return self.__guideapp()
        
        @self.app.route('/punishment')
        def punishment():
            return self.__punishment()
        
        @self.app.route('/tips')
        def tips():
            return self.__tips()
        
        @self.app.route('/detection', methods=['POST', 'GET'])
        def detection():
            return self.__detection()
        
        @self.app.route('/detection/realtime')
        def realtime():
            return self.__realtime()
        
        @self.app.route('/video')
        def video():
            return self.__video()

    def __index(self):
        return render_template('index.html')
    
    def __guideapp(self):
        return render_template('guide.html')

    def __punishment(self):
        return render_template('punishment.html')

    def __tips(self):
        return render_template('tips.html')
    
    def __detection(self):
        predict = None
        if request.method == 'POST':
            img = request.files['face']
            img.save('static/file.jpg', 0)

            image = cv.imread('static/file.jpg', cv.IMREAD_COLOR)
            img_enh = Retinex().img_enh(image)

            faceDetect = cv.CascadeClassifier('support/haarcascade_frontalface_default.xml')
            faces = faceDetect.detectMultiScale(img_enh)
            if len(faces) != 1:
                faces = faceDetect.detectMultiScale(img_enh, 1.3, 5)

            if len(faces) != 0:
                (x, y, w, h) = faces[0]
                cropped = img_enh[y:y + h + 30, x:x + w + 30]
                predict = Video().drowsiness_face(cropped)
            else:
                predict = Video().drowsiness_face(img_enh)

            for x, y, w, h in faces:
                cv.rectangle(img_enh, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv.putText(img_enh, predict, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            cv.imwrite('static/file.jpg', img_enh)

        return render_template('detection.html', predict=predict)

    def __realtime(self):
        return render_template('realtime.html')

    def __gen(self, camera):
        while True:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame +
                b'\r\n\r\n')

    def __video(self):
        return Response(self.__gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def set_config_file(self, total):
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = total

    def run(self):
        self.app.run(debug=True)

def main():
    app = App(__name__)
    app.set_config_file(1)
    app.run()

if __name__ == '__main__':
    main()
