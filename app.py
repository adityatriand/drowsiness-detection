from flask import Flask, render_template, Response, request
from camera import Video
import cv2 as cv

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
faceDetect = cv.CascadeClassifier('support/haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guideapp')
def guideapp():
    return render_template('guide.html')

@app.route('/punishment')
def punishment():
    return render_template('punishment.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/detection', methods=['POST', 'GET'])
def detection():
    predict = None
    if request.method == 'POST':
        img = request.files['face']
        img.save('static/file.jpg', 0)

        image = cv.imread('static/file.jpg', cv.IMREAD_COLOR)
        img_enh = Video().img_enh(image)
        cv.imwrite('static/file.jpg', img_enh)

        faces = faceDetect.detectMultiScale(img_enh, 1.3, 5)
        print(len(faces))
        if len(faces) != 0:
            (x, y, w, h) = faces[0]
            cropped = img_enh[y:y + h+30, x:x + w+30]
            cv.imwrite('static/file.jpg', cropped)

        final_image = cv.imread('static/file.jpg', cv.IMREAD_COLOR)
        predict = Video().drowsiness_face(final_image)
    return render_template('detection.html', predict=predict)

@app.route('/detection/realtime')
def realtime():
    return render_template('realtime.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)