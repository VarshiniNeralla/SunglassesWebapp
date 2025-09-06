from flask import Flask, render_template, Response, request
import cv2
import numpy as np
app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
sunglasses = cv2.imread('static/sunglasses.png', cv2.IMREAD_UNCHANGED)
filter_enabled = True 
def overlay_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sunglass_width = w
        sunglass_height = int(sunglasses.shape[0] * sunglass_width / sunglasses.shape[1])
        resized_sunglasses = cv2.resize(sunglasses, (sunglass_width, sunglass_height))

        y1 = y + int(h / 4)
        y2 = y1 + sunglass_height
        x1 = x
        x2 = x1 + sunglass_width

        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            continue

        if resized_sunglasses.shape[2] == 4:
            alpha_s = resized_sunglasses[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha_s * resized_sunglasses[:, :, c] +
                    alpha_l * frame[y1:y2, x1:x2, c]
                )
        else:
            frame[y1:y2, x1:x2] = resized_sunglasses

    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if filter_enabled:
                frame = overlay_filter(frame)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_filter', methods=['POST'])
def toggle_filter():
    global filter_enabled
    filter_enabled = not filter_enabled
    return {'status': 'ok', 'filter': filter_enabled}

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
