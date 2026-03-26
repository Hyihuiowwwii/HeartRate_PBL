import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from scipy.fft import fft, fftfreq

app = Flask(__name__)

# 1. Initialize Hardware and Models
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables for real-time tracking
signal_buffer = []
latest_bpm = 0

def calculate_bpm(data, fps=30):
    """Applies FFT to extract heart rate frequency."""
    if len(data) < 100:
        return 0
    # Detrend the signal
    normalized_data = np.array(data) - np.mean(data)
    # Apply FFT
    yf = fft(normalized_data)
    xf = fftfreq(len(data), 1/fps)
    # Filter for human heart rate range (0.7Hz to 4Hz)
    idx = np.where((xf >= 0.7) & (xf <= 4.0))
    if len(idx[0]) == 0: return 0
    peak_freq = xf[idx][np.argmax(np.abs(yf[idx]))]
    return abs(int(peak_freq * 60))

def generate_frames():
    global latest_bpm
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # 2. ROI Selection: Forehead region for rPPG
            # Coordinates: top 20% of the detected face box
            roi = frame[y+int(h*0.1):y+int(h*0.25), x+int(w*0.3):x+int(w*0.7)]
            
            if roi.size > 0:
                # 3. Extract Green Channel average
                avg_green = np.mean(roi[:, :, 1])
                signal_buffer.append(avg_green)
                
                # Keep 5-10 seconds of data for FFT
                if len(signal_buffer) > 200:
                    signal_buffer.pop(0)
                    latest_bpm = calculate_bpm(signal_buffer)
            
            # Visual feedback for the evaluator
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 123, 255), 2)
            cv2.putText(frame, f"BPM: {latest_bpm if latest_bpm > 0 else 'Calculating...'}", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    # Sends real-time data to the JavaScript graph in dashboard.html
    return jsonify({
        "bpm": latest_bpm,
        "signal": signal_buffer[-30:] if len(signal_buffer) > 0 else []
    })

if __name__ == "__main__":
    app.run(debug=True)
