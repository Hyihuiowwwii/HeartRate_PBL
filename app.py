from flask import Flask, render_template, Response, jsonify
from rppg_engine import RPPGEngine
import cv2

app = Flask(__name__)
engine = RPPGEngine()
camera = cv2.VideoCapture(0)
latest_bpm = 0

def generate():
    global latest_bpm
    while True:
        success, frame = camera.read()
        if not success: break
        frame, bpm = engine.process_frame(frame)
        latest_bpm = bpm
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    return jsonify({"bpm": latest_bpm, "signal": engine.buffer[-20:]})

if __name__ == "__main__": app.run(debug=True)
