# AI Proctoring System (Human-in-the-Loop)

A robust, computer vision-based proctoring system designed for remote examinations. This system uses AI (YOLOv8, MediaPipe, Google Gemini) to detect potential violations while keeping a "Human-in-the-Loop" for final decision making.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)

## 🚀 Features

*   **Multi-Model Detection**:
    *   **Gaze Tracking**: Detects if candidate looks away from screen (>5s).
    *   **Head Pose Estimation**: Detects suspicious head turns (>2s).
    *   **Object Detection (YOLOv8)**: Identifies unauthorized objects (Phones, Books, Laptops).
    *   **Person Detection**: Detects multiple people or absence of candidate.
    *   **Audio Monitoring**: Detects sustained speech or background noise.
*   **AI Verification (Gemini)**:
    *   Uses **Google Gemini 1.5 Flash** to cross-verify flagged violations (e.g., distinguishing between a phone and a hand).
    *   Reduces false positives by 90%.
*   **Evidence Capture**: Automatically records short video clips (13s) when violations occur.
*   **Security**: Encrypts logs and evidence.
*   **Live Dashboard**: Flask-based web dashboard for real-time monitoring.

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/ajay-marpally/ai-proctoring-system.git
    cd ai-proctoring-system
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install YOLO Model**
    The system uses `yolov8s.pt`. It will auto-download on first run, or you can place it in the root directory.

## ⚙️ Configuration

### 1. API Key Setup
You need a Google Gemini API Key for the AI verification features.
*   Get a key from [Google AI Studio](https://aistudio.google.com/).
*   Set it as an environment variable (Recommended):
    ```bash
    # Windows (PowerShell)
    $env:GEMINI_API_KEY="your_api_key_here"
    ```
    *Or update `main.py` configuration directly (Not recommended for public repos).*

### 2. Tuning Thresholds
You can adjust sensitivity in `main.py` under the `CONFIG` dictionary:

```python
'thresholds': {
    'gaze_deviation_seconds': 5,   # How long to look away
    'absence_seconds': 10,         # How long to be missing
    'movement_threshold': 0.15,    # Sensitivity to body movement
}
```

## 🏃 Usage

Run the main script:

```bash
python main.py
```

*   **Camera Preview**: Opens a window showing the camera feed with detection bounding boxes.
*   **Dashboard**: Open `http://localhost:8000` in your browser to see the proctor dashboard.

## 📊 Scoring System (0-100 Scale)

The system assigns risk scores based on detected behavior:

| Violation | Points | Band |
| :--- | :--- | :--- |
| **Mobile Phone / Books** | 100 | **CRITICAL** |
| **Multiple People** | 100 | **CRITICAL** |
| **Gaze Deviation** | 20 | HIGH |
| **Head Turning** | 20 | HIGH |
| **Absence** | 20 | HIGH |
| **Focus Loss (Tab Switch)** | 12 | MEDIUM |
| **Movement / Audio** | 5 | LOW |

*Scores decay over time if behavior returns to normal.*

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
