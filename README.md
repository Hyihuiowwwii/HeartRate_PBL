# HeartRate Pro — Contactless Monitoring (PBL Phase 2)

## Project Overview
This project uses **Remote Photoplethysmography (rPPG)** to measure heart rate through a standard webcam. It detects subtle color changes in the skin caused by blood flow.

## Tech Stack
* **Python 3.x**: Core programming language.
* **Flask**: Web framework for the user interface.
* **OpenCV**: Computer vision for face detection and ROI extraction.
* **NumPy/SciPy**: Signal processing and FFT analysis.

## Current Progress (Phase 2: 50% Completion)
- [x] Initial Flask server setup.
- [x] Real-time webcam feed integration.
- [x] Face detection using Haar Cascades.
- [x] Forehead ROI (Region of Interest) isolation.
- [x] Green channel signal extraction logic.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python app.py`
3. Open `http://127.0.0.1:5000` in your browser.
