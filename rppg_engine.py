import cv2
import numpy as np
from scipy.fft import fft, fftfreq

class RPPGEngine:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.buffer = []
        self.buffer_size = 150 # 5 seconds at 30 fps

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        bpm = 0
        
        for (x, y, w, h) in faces:
            # ROI: Forehead (Top 20% of face)
            roi = frame[y+int(h*0.1):y+int(h*0.25), x+int(w*0.3):x+int(w*0.7)]
            if roi.size > 0:
                green_val = np.mean(roi[:, :, 1]) # Green Channel Extraction
                self.buffer.append(float(green_avg))
                
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
                    bpm = self.calculate_bpm()
            
            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 123, 255), 2)
        return frame, int(bpm)

    def calculate_bpm(self):
        # Detrend and FFT
        data = np.array(self.buffer) - np.mean(self.buffer)
        fft_out = np.abs(fft(data))
        freqs = fftfreq(len(data), 1/30)
        
        # Filter for human heart rate (0.7Hz - 4Hz)
        val_idx = np.where((freqs >= 0.7) & (freqs <= 4.0))
        peak_freq = freqs[val_idx][np.argmax(fft_out[val_idx])]
        return peak_freq * 60
