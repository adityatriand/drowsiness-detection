from flask import Flask, render_template, Response, jsonify
from camera import Video

app = Flask(__name__)
predict = -1

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        result = camera.get_frame()
        frame = result[0]
        global predict
        predict = result[1]
        print(predict)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    global predict
    if predict == 0:
        status = 'mengantuk'
    elif predict == 1:
        status = 'tidak mengantuk'
    else:
        status = 'tidak ada wajah'
    data = {'predict': status}
    return jsonify(data)

app.run(debug=True)