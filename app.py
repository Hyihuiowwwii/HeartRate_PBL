import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)
# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # ROI: Focus on the forehead for the best rPPG signal
            forehead_roi = frame[y+int(h/10):y+int(h/4), x+int(w/4):x+int(3*w/4)]
            
            # Extract Green Channel average (most sensitive to heart rate)
            avg_green = np.mean(forehead_roi[:, :, 1])
            
            # UI Overlay for Evaluator
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Signal: {avg_green:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return "<h1>HeartRate Pro: Live Monitoring</h1><img src='/video_feed' width='800'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
