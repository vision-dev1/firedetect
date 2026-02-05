# 🔥 Fire Detection Tool

Real-time fire detection system using computer vision and audio alerts.

## 📋 Project Overview

This tool detects fire in real-time using your webcam and plays an alarm sound when fire is detected. It uses computer vision techniques to identify fire-like colors (red/orange hues) and confirms fire presence based on area and intensity thresholds.

## ✨ Features

- Real-time fire detection using webcam
- Color-based segmentation for fire identification
- Morphological operations to reduce false positives
- Continuous alarm sound when fire is detected
- Instant alarm stop when fire disappears
- Bounding boxes around detected fire regions

## 📁 Project Structure

```
firedetect/
│
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── sounds/
│   └── alert.mp3        # Alarm sound file (originally named alarm.mp3)
└── README.md            # This file
```

## ⚙️ Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the fire detection tool:
```bash
python main.py
```

Controls:
- Press `Q` to quit the application

## 🧠 How It Works

1. Captures video frames from the webcam
2. Converts frames to HSV color space
3. Applies color segmentation to detect red/orange regions
4. Uses morphological operations to reduce noise
5. Analyzes contours to identify potential fire regions
6. Confirms fire based on area and intensity thresholds
7. Plays alarm sound when fire is confirmed
8. Stops alarm immediately when fire disappears

## 🎯 Requirements

- Python 3.x
- OpenCV
- NumPy
- Pygame

## 📝 Notes

- Make sure your webcam is properly connected
- The alarm sound file (alert.mp3) must be in the `sounds/` directory
- Adjust `fire_threshold` and `fire_intensity_threshold` in `main.py` if needed
- The actual sound file may be named `alert.mp3` instead of `alarm.mp3` depending on your version
- Designed by Vision | Made By Vision | github.com/vision-dev1

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[GitHub](https://github.com/vision-dev1) <br>
[Portfolio](https://visionkc.com.np)
