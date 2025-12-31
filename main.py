import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import os
import pyautogui
import time
import threading
from collections import deque, Counter
from cryptography.fernet import Fernet
import requests
import yaml
from flask import Flask, render_template_string
import queue
from ultralytics import YOLO
import torch
from silero_vad import load_silero_vad
import json
import base64
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ========================
# Configuration
# ========================
CONFIG = {
    'thresholds': {
        # SCORING & BANDS
        'points': {
            "multiple_faces": 100,        
            "phone_detected": 100,         
            "book_detected": 100,          
            "unauthorized_login": 100,     
            "gaze_deviation": 20,         
            "head_orientation_issue": 20,  
            "suspicious_object": 20,       
            "candidate_absent": 20,        
            "focus_loss": 12,              
            "audio_spike": 5,             
            "excessive_movement": 5,       
        },
        'risk_bands': {
            "clear": 0,        
            "low": 5,          
            "medium": 12,      
            "high": 20,        
            "critical": 100    
        },

        # DURATION THRESHOLDS (seconds)
        'gaze_deviation_seconds': 5,
        'absence_seconds': 10,
        'absence_grace_period': 5,
        'audio_sustained_seconds': 3,
        'head_turn_buffer_seconds': 2,

        # RISK DECAY
        'decay_amount_per_sec': 5,  
        
        # SENSOR PARAMETERS
        'head_yaw_threshold': 20,       
        'head_pitch_threshold': 15,     # Degrees
        'gaze_threshold_left': 0.40,    
        'gaze_threshold_right': 0.65,   
        'gaze_threshold_top': 0.35,     
        'gaze_threshold_bottom': 0.75,  
    },
    'detection': {
        'frame_interval_fps': 10,
        'buffer_max_history_seconds': 30
    },
    'objects': {
        'model_path': 'yolov8s.pt',
        'confidence_threshold': 0.5,
        'phone_labels': ['cell phone', 'mobile', 'phone', 'smartphone'],
        'book_labels': ['book', 'notebook', 'paper', 'magazine'],
        'suspicious_labels': ['laptop', 'tablet', 'monitor', 'tv']
    },
    'evidence': {
        'clip_duration_seconds': 10, 
        'cooldown_seconds': 15,
    },
    'gemini': {
        'api_key': os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE'), # Load from environment for security
        'enabled': True,
        'model_name': 'gemini-2.5-flash-lite'
    }
}

# ========================
# Log queue for dashboard
# ========================
log_queue = queue.Queue()

def log_message(message):
    print(message)
    log_queue.put(message)

# Dashboard state
evidence_log = []
last_action = None
current_risk_for_dashboard = 0.0
current_risk_band_for_dashboard = "clear"
session_risk_for_dashboard = 0.0
session_band_for_dashboard = "clear"
recent_log_lines = []
critical_flags = []  # NEW: Track flags for human review

# ========================
# Encryption helper
# ========================
class Encryptor:
    def __init__(self, key=None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)

# ========================
# Rolling Buffer
# ========================
class RollingBuffer:
    def __init__(self, max_seconds=15, fps=10):
        self.maxlen = max_seconds * fps
        self.frames = deque(maxlen=self.maxlen)
        self.timestamps = deque(maxlen=self.maxlen)

    def add_frame(self, frame):
        self.frames.append(frame.copy())
        self.timestamps.append(time.time())

    def get_clip(self, duration_seconds=15, target_timestamp=None):
        """
        Retrieves a clip based on duration or centered around a target timestamp.
        If target_timestamp is None, takes the most recent 'duration_seconds'.
        """
        now = time.time()
        if target_timestamp:
            # Capture 5 seconds BEFORE target and 8 seconds AFTER target (Total 13s)
            cutoff_start = target_timestamp - 5
            cutoff_end = target_timestamp + 8
        else:
            cutoff_start = now - duration_seconds
            cutoff_end = now + 1.0 # Buffer for current frame sync
            
        recent_frames = []
        recent_ts = []
        for ts, frame in zip(self.timestamps, self.frames):
            if ts >= cutoff_start and ts <= cutoff_end:
                recent_frames.append(frame)
                recent_ts.append(ts)
        return recent_frames, recent_ts

# ========================
# Visual Detection
# ========================
class VisualDetector:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        self.last_pose_landmarks = None
        # ADJUSTED: Tuned to 0.18 (Middle ground). 0.25 was too high, 0.12 too low.
        self.movement_threshold = 0.15
        # ADJUSTED: Reduced buffer from 5 to 3 for faster detection (0.3s)
        self.movement_buffer = deque(maxlen=3)
        self.face_count_buffer = deque(maxlen=15) # Buffer for face detection noise
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5
        )
        self.yolo = YOLO(CONFIG['objects']['model_path'])
        
        # FPS and Timers
        self.fps = CONFIG['detection']['frame_interval_fps']

        # 3D Model Points for Head Pose Estimation (Generic Face)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right Mouth corner
        ])

        self.last_face_present_ts = time.time()
        self.last_gaze_ok_ts = time.time()
        self.gaze_consecutive_count = 0
        
        # Sustained threshold: Gaze/Head must deviate for X seconds
        self.gaze_target_frames = CONFIG['thresholds']['gaze_deviation_seconds'] * self.fps
        self.head_turn_consecutive_count = 0 
        self.head_turn_target_frames = CONFIG['thresholds']['head_turn_buffer_seconds'] * self.fps
        
        self.debug_frame_count = 0

    def get_head_pose(self, image_points, size):
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0, 0, 0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return pitch, yaw, roll


        
    def process_frame(self, frame):
        self.debug_frame_count += 1
        now = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        face_results = self.face_detection.process(rgb_frame)
        raw_num_faces = len(face_results.detections) if face_results.detections else 0
        self.face_count_buffer.append(raw_num_faces)
        
        # Buffer logic: Only flag multiple faces if detected in > 40% of recent frames
        if len(self.face_count_buffer) >= 5:
            count_more_than_one = sum(1 for c in self.face_count_buffer if c > 1)
            multiple_faces = count_more_than_one > (len(self.face_count_buffer) * 0.4)
        else:
            multiple_faces = False
            
        face_present = raw_num_faces >= 1

        if face_present:
            self.last_face_present_ts = now

        # Gaze detection
        mesh_results = self.face_mesh.process(rgb_frame)
        gaze_deviation_instant = False

        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark

            left_iris = landmarks[468]
            right_iris = landmarks[473]
            left_eye_outer = landmarks[33]
            left_eye_inner = landmarks[133]
            right_eye_outer = landmarks[263]
            right_eye_inner = landmarks[362]

            left_eye_top = landmarks[159]
            left_eye_bot = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bot = landmarks[374]

            left_width = left_eye_inner.x - left_eye_outer.x + 1e-6
            right_width = right_eye_outer.x - right_eye_inner.x + 1e-6
            left_height = left_eye_bot.y - left_eye_top.y + 1e-6
            right_height = right_eye_bot.y - right_eye_top.y + 1e-6

            left_gaze_ratio_x = (left_iris.x - left_eye_outer.x) / left_width
            right_gaze_ratio_x = (right_eye_outer.x - right_iris.x) / right_width
            avg_gaze_ratio_x = (left_gaze_ratio_x + right_gaze_ratio_x) / 2.0

            left_gaze_ratio_y = (left_iris.y - left_eye_top.y) / left_height
            right_gaze_ratio_y = (right_iris.y - right_eye_top.y) / right_height
            avg_gaze_ratio_y = (left_gaze_ratio_y + right_gaze_ratio_y) / 2.0
            
            # DEBUG: Visualize gaze
            if self.debug_frame_count % 10 == 0:
                print(f"DEBUG GAZE: X={avg_gaze_ratio_x:.2f}, Y={avg_gaze_ratio_y:.2f}")

            # USER REQUEST: 0.35 - 0.65 (Symmetric, Strict but Reasonable)
            # UPDATED: Using Configurable Thresholds
            t_left = CONFIG['thresholds']['gaze_threshold_left']
            t_right = CONFIG['thresholds']['gaze_threshold_right']
            t_top = CONFIG['thresholds']['gaze_threshold_top']
            t_bot = CONFIG['thresholds']['gaze_threshold_bottom']
            
            iris_deviation_x = avg_gaze_ratio_x < t_left or avg_gaze_ratio_x > t_right
            iris_deviation_y = avg_gaze_ratio_y < t_top or avg_gaze_ratio_y > t_bot
            iris_deviation = iris_deviation_x or iris_deviation_y
            
            # HEAD POSE ESTIMATION (Yaw check)
            head_orientation_issue = False
            image_h, image_w, _ = rgb_frame.shape
            
            # Mapping MediaPipe landmarks to 3D model points
            # 1 (Nose), 152 (Chin), 263 (Left Eye), 33 (Right Eye), 291 (Left Mouth), 61 (Right Mouth)
            face_3d_points = []
            for idx in [1, 152, 263, 33, 291, 61]:
                lm = landmarks[idx]
                x, y = int(lm.x * image_w), int(lm.y * image_h)
                face_3d_points.append([x, y])
            
            face_3d_points = np.array(face_3d_points, dtype="double")
            
            pitch, yaw, roll = self.get_head_pose(face_3d_points, (image_h, image_w))
            
            # Check Yaw (Left/Right) and Pitch (Up/Down)
            yaw_threshold = CONFIG['thresholds']['head_yaw_threshold']
            pitch_threshold = CONFIG['thresholds']['head_pitch_threshold']
            
            is_turning = abs(yaw) > yaw_threshold or abs(pitch) > pitch_threshold
            
            # DEBUG: Print Pose occasionally
            if self.debug_frame_count % 10 == 0:
                print(f"DEBUG HEAD: Yaw={yaw:.1f}, Pitch={pitch:.1f}")

            if is_turning:
                self.head_turn_consecutive_count += 1
                if self.head_turn_consecutive_count >= self.head_turn_target_frames:
                    head_orientation_issue = True
            else:
                self.head_turn_consecutive_count = max(0, self.head_turn_consecutive_count - 2) # Fast decay
                head_orientation_issue = False

            # Calculate direction labels for preview
            head_label = "HEAD: CENTER"
            if yaw > yaw_threshold: head_label = "HEAD: LEFT"
            elif yaw < -yaw_threshold: head_label = "HEAD: RIGHT"
            elif pitch > pitch_threshold: head_label = "HEAD: DOWN"
            elif pitch < -pitch_threshold: head_label = "HEAD: UP"

            gaze_label = "GAZE: CENTER"
            if avg_gaze_ratio_x < t_left: gaze_label = "GAZE: LEFT"
            elif avg_gaze_ratio_x > t_right: gaze_label = "GAZE: RIGHT"
            elif avg_gaze_ratio_y < t_top: gaze_label = "GAZE: UP"
            elif avg_gaze_ratio_y > t_bot: gaze_label = "GAZE: DOWN"

            instant_deviation = iris_deviation or head_orientation_issue

            if instant_deviation:
                self.gaze_consecutive_count += 1
                if self.gaze_consecutive_count >= self.gaze_target_frames:
                    gaze_deviation_instant = True
            else:
                self.gaze_consecutive_count = max(0, self.gaze_consecutive_count - 3)
                gaze_deviation_instant = False
        else:
            # FIX: If Face Detected but Mesh Lost, it usually means extreme angle
            # Don't reset to 0! Treat it as Deviation if face is present.
            if face_present:
                 self.gaze_consecutive_count += 1
                 if self.gaze_consecutive_count >= self.gaze_target_frames:
                     gaze_deviation_instant = True
                 
                 # NEW: Also assume head turn if face is present (but mesh lost due to angle)
                 self.head_turn_consecutive_count += 1
                 if self.head_turn_consecutive_count >= self.head_turn_target_frames:
                     head_orientation_issue = True
            else:
                 self.gaze_consecutive_count = 0
                 self.head_turn_consecutive_count = 0 
            
            # Defaults if mesh lost
            pitch, yaw, roll = 0.0, 0.0, 0.0
            head_label = "HEAD: UNKNOWN" if face_present else "HEAD: ABSENT"
            gaze_label = "GAZE: UNKNOWN" if face_present else "GAZE: ABSENT"
            avg_gaze_ratio = 0.5
            
            # Reset head orientation if no face mesh (handled by face_present logic above mostly)
            # head_orientation_issue = False <--- REMOVED (Handled above now) 

        gaze_deviation = gaze_deviation_instant
        if gaze_deviation:
             print("DEBUG GAZE: Eyes looking away!")
        
        # DEBUG: Print Ratio occasionally to help user calibrate
        if self.debug_frame_count % 20 == 0 and face_present:
             pass 
             # print(f"DEBUG GAZE RATIO: {avg_gaze_ratio:.2f} (Left<0.25, Right>0.75)")
             # We will enable this if needed, but for now just the event

        # Absence detection
        absence_duration = now - self.last_face_present_ts
        grace_period = CONFIG['thresholds']['absence_grace_period']
        candidate_absent = (
            absence_duration >= (CONFIG['thresholds']['absence_seconds'] + grace_period)
        )

        # Movement detection
        pose_results = self.pose.process(rgb_frame)
        excessive_movement = False

        if pose_results.pose_landmarks:
            current = pose_results.pose_landmarks.landmark
            if self.last_pose_landmarks:
                idxs = [0, 11, 12, 23, 24]
                diffs = []
                for i in idxs:
                    dx = current[i].x - self.last_pose_landmarks[i].x
                    dy = current[i].y - self.last_pose_landmarks[i].y
                    diffs.append((dx * dx + dy * dy) ** 0.5)
                avg_movement = sum(diffs) / len(diffs)
                
                # DEBUG: Visualize movement score
                if self.debug_frame_count % 10 == 0:
                     print(f"DEBUG MOVE: {avg_movement:.3f} (Thresh: {self.movement_threshold})")

                self.movement_buffer.append(avg_movement > self.movement_threshold)
                
                if len(self.movement_buffer) >= 3:
                     # Requiring 2 out of 3 frames to be flagged
                    excessive_movement = sum(self.movement_buffer) >= 2
            self.last_pose_landmarks = current
        else:
            excessive_movement = False
            self.last_pose_landmarks = None

        # Object detection
        phone_detected = False
        book_detected = False
        suspicious_object_detected = False
        unauthorized_object = False

        suspicious_object_detected = False
        unauthorized_object = False

        try:
            results = self.yolo.predict(
                frame, 
                imgsz=640, 
                conf=CONFIG['objects']['confidence_threshold'],
                verbose=False
            )
            
            phone_labels = [l.lower() for l in CONFIG['objects']['phone_labels']]
            book_labels = [l.lower() for l in CONFIG['objects']['book_labels']]
            suspicious_labels = [l.lower() for l in CONFIG['objects']['suspicious_labels']]
            
            for r in results:
                for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    if conf < CONFIG['objects']['confidence_threshold']:
                        # DEBUG: Print what we see even if low confidence
                        # print(f"DEBUG YOLO: Ignored {r.names[int(cls_id)]} ({conf:.2f})")
                        continue
                    
                    label = r.names[int(cls_id)].lower()
                    
                    # DEBUG: Explicitly print if specific items are found
                    if label in book_labels or label in phone_labels or label in suspicious_labels:
                        print(f"DEBUG YOLO: Detected {label} ({conf:.2f})")
                    if label in phone_labels:
                        phone_detected = True
                    if label in book_labels:
                        book_detected = True
                    if label in suspicious_labels:
                        suspicious_object_detected = True
        except Exception as e:
            # Prevent thread death if YOLO crashes
            print(f"ERROR: YOLO Crash ignored: {e}")
            pass

        unauthorized_object = phone_detected or book_detected or suspicious_object_detected

        return {
            "face_present": face_present,
            "multiple_faces": multiple_faces,
            "candidate_absent": candidate_absent,
            "gaze_deviation": gaze_deviation,
            "head_orientation_issue": head_orientation_issue if 'head_orientation_issue' in locals() else False,
            "excessive_movement": excessive_movement,
            "phone_detected": phone_detected,
            "book_detected": book_detected,
            "suspicious_object": suspicious_object_detected,
            "unauthorized_object": unauthorized_object,
            # NEW: Detailed movement data for preview
            "yaw": yaw if 'yaw' in locals() else 0.0,
            "pitch": pitch if 'pitch' in locals() else 0.0,
            "avg_gaze_ratio": avg_gaze_ratio if 'avg_gaze_ratio' in locals() else 0.5,
            "head_label": head_label if 'head_label' in locals() else "HEAD: ABSENT",
            "gaze_label": gaze_label if 'gaze_label' in locals() else "GAZE: ABSENT"
        }

# ========================
# Audio Monitor
# ========================
class AudioMonitor:
    def __init__(self, callback):
        self.callback = callback
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.running = False

        self.sample_rate = 16000
        self.chunk_size = 512

        self.vad_model = load_silero_vad()
        print("DEBUG: Silero VAD Loaded", flush=True)
        
        self.speech_state = False
        self.speech_start_ts = None
        self.silence_start_ts = None  # NEW: For grace period
        
        self.suspicious_seconds = CONFIG['thresholds']['audio_sustained_seconds']
        self.silence_grace_seconds = 1.0  # Allow 1s silence before resetting speech
        self.speech_pattern_buffer = deque(maxlen=50)
        self.sustained_speech_threshold = 0.70
        self.debug_frame_count = 0  # FIX: Initialize counter

    def start(self):
        def _callback(in_data, frame_count, time_info, status):
            self.debug_frame_count += 1
            audio_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            if audio_np.size == 0:
                speech_prob = 0.0
                speech_ratio = 0.0
            else:
                audio_np /= 32768.0
                if audio_np.size > 512:
                    audio_np = audio_np[:512]
                elif audio_np.size < 512:
                    audio_np = np.pad(audio_np, (0, 512 - audio_np.size))
                audio_tensor = torch.from_numpy(audio_np)
                speech_prob = float(self.vad_model(audio_tensor, self.sample_rate).detach())

            now = time.time()
            audio_spike = False

            now = time.time()
            audio_spike = False

            # DEBUG: Check if mic is working
            # if speech_prob > 0.1:
            #     print(f"DEBUG AUDIO: Level={speech_prob:.2f}")

            if speech_prob > 0.6:
                if self.debug_frame_count % 10 == 0:
                     print(f"DEBUG AUDIO: Speech Detected (Prob: {speech_prob:.2f})")
                self.speech_pattern_buffer.append(1)
                
                if not self.speech_state:
                    self.speech_state = True
                    self.speech_start_ts = now
                self.silence_start_ts = None # Reset silence timer
            else:
                self.speech_pattern_buffer.append(0)
                
                # GRACE PERIOD LOGIC
                if self.speech_state:
                    if self.silence_start_ts is None:
                        self.silence_start_ts = now
                    elif now - self.silence_start_ts > self.silence_grace_seconds:
                        # Grace period over, reset
                        self.speech_state = False
                        self.speech_start_ts = None
                        self.silence_start_ts = None
                else:
                    self.speech_state = False
                    self.speech_start_ts = None

            speech_ratio = 0.0
            if len(self.speech_pattern_buffer) >= 40:
                speech_ratio = sum(self.speech_pattern_buffer) / len(self.speech_pattern_buffer)
                
                if speech_ratio > self.sustained_speech_threshold:
                    if self.speech_start_ts and (now - self.speech_start_ts) >= self.suspicious_seconds:
                        audio_spike = True

            data = {
                "audio_spike": audio_spike,
                "speech_probability": speech_prob,
                "speech_ratio": speech_ratio
            }
            self.callback(data)
            return (in_data, pyaudio.paContinue)

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=_callback
        )
        self.stream.start_stream()
        self.running = True

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# ========================
# System Monitor
# ========================
class SystemMonitor:
    def __init__(self):
        self.last_window = None
        self.tab_switch_count = 0

    def check_focus_loss(self):
        current_window = pyautogui.getActiveWindowTitle()
        if self.last_window and current_window != self.last_window:
            self.tab_switch_count += 1
            self.last_window = current_window
            return True
        self.last_window = current_window
        return False



# ========================
# Hybrid Verifier (Gemini 1.5 Flash)
# ========================
class HybridVerifier:
    def __init__(self, config):
        self.config = config
        self.enabled = config['gemini']['enabled']
        self.api_key = config['gemini']['api_key']
        self.last_call_time = 0
        self.min_interval = 4.0 # Limit to ~15 calls per minute (Free tier)
        self.lock = threading.Lock()
        
        if self.enabled and self.api_key:
            model_name = config['gemini']['model_name']
            # Ensure model_name doesn't already have 'models/' prefix
            path_name = model_name if model_name.startswith('models/') else f"models/{model_name}"
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{path_name}:generateContent?key={self.api_key}"
            log_message(f"✅ HybridVerifier: Ready (REST Mode) | Model: {model_name}")
        else:
            self.enabled = False

    def verify_frame_async(self, frame_cv2, detection_type, callback_success, callback_fail):
        if not self.enabled:
            return

        # Rate Limiting
        with self.lock:
            now = time.time()
            if now - self.last_call_time < self.min_interval:
                # Too soon, skip verification (assume local detection was wrong/noisy to be safe? 
                # OR assume it was right? For proctoring, usually false positives are annoying.
                # Let's skip verification and NOT flag it as critical yet.)
                return
            self.last_call_time = now

        # Run in separate thread
        threading.Thread(target=self._worker, args=(frame_cv2, detection_type, callback_success, callback_fail)).start()

    def _worker(self, frame_cv2, detection_type, callback_success, callback_fail):
        try:
            # Encode image to Base64
            # Resize for faster upload (optional, but good for stability)
            h, w = frame_cv2.shape[:2]
            if w > 1024:
                scale = 1024 / w
                frame_cv2 = cv2.resize(frame_cv2, (0, 0), fx=scale, fy=scale)
            
            _, buffer = cv2.imencode('.jpg', frame_cv2)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            # Detailed Prompt Strategy for Deep Electronics Scanning
            prompt_text = (
                f"You are a highly vigilant AI Exam Proctor. Analyze this image for academic dishonesty.\n"
                f"BACKGROUND: The local system flagged: {detection_type}. However, local detection can be inaccurate (e.g., mistaking earbuds for phones).\n\n"
                f"TASK: \n"
                f"1. Perform a DEEP SCAN for hidden devices: earbuds (eardopes/airpods), smartwatches, charging cases, or earpieces.\n"
                f"2. Confirm if the original flag '{detection_type}' is correct. If not, IDENTIFY what it actually is.\n"
                f"3. Check for other violations: unauthorized papers, other people, or candidate looking away from the screen area.\n\n"
                f"RESULT FORMAT:\n"
                f"If any violation is found, reply: 'VIOLATION: <Object Name> - <1-2 short sentences describing why it is a violation>'.\n"
                f"Example: 'VIOLATION: EARBUDS - The candidate is wearing white wireless earbuds in both ears.'\n"
                f"Example: 'VIOLATION: SMART WATCH - A digital watch is visible on the student's left wrist.'\n"
                f"If clean, reply 'CLEAR'."
            )
            
            # Construct JSON Payload for Gemini API
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt_text},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }}
                    ]
                }]
            }
            
            headers = {'Content-Type': 'application/json'}
            
            # Make REST Request
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=12)
            
            if response.status_code == 200:
                data = response.json()
                try:
                    # Log the full response for debugging if needed (rare)
                    # print(f"DEBUG: Gemini Raw Response: {json.dumps(data)}")
                    result_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Gemini Parse Error: {e} | Raw: {data}")
                    result_text = "ERROR: Unexpected API response structure"
            else:
                try:
                    err_details = response.json()
                except:
                    err_details = response.text
                print(f"ERROR: Gemini API Error {response.status_code}: {err_details}")
                callback_fail(detection_type, f"API Error {response.status_code}")
                return

            # Parse Result
            print(f"🤖 GEMINI RESPONSE for {detection_type}: {result_text[:100]}...") # Log start of response
            
            if "VIOLATION:" in result_text.upper():
                # Extract description
                description = result_text.split(":", 1)[1].strip() if ":" in result_text else "Confirmed by AI"
                callback_success(detection_type, description)
            else:
                callback_fail(detection_type, result_text)

        except Exception as e:
            print(f"ERROR: Gemini REST API Failed: {e}")
            callback_fail(detection_type, str(e))

# ========================
# Rule Engine 
# ========================
class RuleEngine:
    """
    Human-in-the-Loop Model:
    - AI detects and flags violations
    - Generates risk scores and alerts
    - Human proctor makes final decisions
    - NO auto-termination
    """

    def __init__(self, config):
        self.config = config
        self.VIOLATION_POINTS = config['thresholds']['points']
        self.RISK_BANDS = config['thresholds']['risk_bands']
        
        self.risk_score = 0.0      # Current Risk (decays fast)
        self.session_risk_score = 0.0 # Session Risk (max reached)
        
        self.last_decay_time = time.time()
        self.last_evidence_time = None
        self.last_focus_loss_time = 0

        # Track active violations
        self.active_critical_violations = {} 
        self.violation_history = [] 
        
        self.last_warn_notification = None
        self.last_pause_notification = None
        
        self.decay_amount = config['thresholds']['decay_amount_per_sec']
        
        # Hybrid Verifier
        self.verifier = HybridVerifier(config)
        self.pending_verifications = {} 
        self.confirmed_violations = {}  
        self.trigger_confirmed_evidence = None 

    def _callback_verified(self, v_key, reason):
        """Called when Gemini returns a VIOLATION"""
        # KEYWORD PROMOTION: Scan the reason for specific hidden keywords
        # This allows AI to "discover" a different cheat than what was requested.
        effective_key = v_key.lower()
        search_text = reason.lower()
        
        # Priority mapping: EARBUDS/PERSON/BOOK/PHONE
        if any(kw in search_text for kw in ["earbud", "earphone", "eardope", "airpod", "headphone"]):
            effective_key = "unauthorized electronic"
        elif any(kw in search_text for kw in ["person", "people", "human", "face"]):
            effective_key = "multiple people"
        elif any(kw in search_text for kw in ["book", "paper", "note", "page"]):
            effective_key = "book/notes"
        elif "phone" in search_text:
            effective_key = "mobile phone"
            
        log_message(f"✅ AI CONFIRMED: {effective_key} ({reason})")
        
        # Retrieve the original start time from pending
        start_time = time.time() 
        if v_key.lower() in self.pending_verifications:
            start_time = self.pending_verifications[v_key.lower()]
            del self.pending_verifications[v_key.lower()]

        self.confirmed_violations[effective_key] = {
            "time": time.time(),
            "start_time": start_time, 
            "reason": reason
        }
        self.trigger_confirmed_evidence = {
            "reason": f"{effective_key}_{reason[:20]}",
            "start_time": start_time
        }

    def _callback_rejected(self, v_key, reason):
        """Called when Gemini REJECTS a violation (Verdict: CLEAN)"""
        log_message(f"🤖 AI CROSS-CHECK: [REJECTED] {v_key}. Reason: {reason}")
        if v_key.lower() in self.pending_verifications:
            del self.pending_verifications[v_key.lower()]
        
        # Keep minor points to indicate some activity, but not enough for High/Critical
        self.risk_score = max(self.risk_score, 5.0)  # Changed from 50.0 (0-100 scale) 
        
    def _apply_decay(self):
        """Decay risk based on config"""
        now = time.time()
        if now - self.last_decay_time >= 1.0:
            self.risk_score = max(0, self.risk_score - self.decay_amount)
            self.last_decay_time = now
            
            # Clean up old critical violations
            expired_types = []
            for vtype, timestamp in self.active_critical_violations.items():
                if now - timestamp > 5:  # Fast clear from active list
                    expired_types.append(vtype)
            for vtype in expired_types:
                del self.active_critical_violations[vtype]
    
    def process_detection(self, detection_data, frame=None):
        """Process detection and generate flags for human review"""
        now = time.time()
        self._apply_decay()

        violations = []
        score_increase = 0
        is_critical_frame = False
        critical_types_this_frame = []

        # CLEANUP Confirmed Violations (expire after 15s for better visibility)
        for vtype in list(self.confirmed_violations.keys()):
            if now - self.confirmed_violations[vtype]["time"] > 15.0:
                 del self.confirmed_violations[vtype]

        # ---------------------------
        # SECOND OPINION (Gemini AI)
        # ---------------------------
        
        # User Request: Revalidate High and Critical only
        check_targets = []
        if detection_data.get("phone_detected"): check_targets.append("mobile phone")
        if detection_data.get("book_detected"): check_targets.append("book/notes")
        if detection_data.get("multiple_faces"): check_targets.append("multiple people")
        if detection_data.get("suspicious_object"): check_targets.append("secondary device")
        
        # Behavioral targets (Gaze/Head/Absence) are now handled by Local
        # for maximum reliability and zero-latency response.

        for target in check_targets:
            v_key = target.lower()
            
            # CHECK FOR KEYWORD PROMOTIONS (Fix for Earbuds/Phone mismatch)
            keys_to_check = [v_key]
            if "phone" in v_key: keys_to_check.append("unauthorized electronic")
            
            found_key = None
            for k in keys_to_check:
                if k in self.confirmed_violations:
                    found_key = k
                    break

            if found_key:
                 # AI Confirmed this! (Agreed with Local OR Discovered new item)
                 details = self.confirmed_violations[found_key]
                 
                 # Map callback keys to violation types
                 v_type = "suspicious_object" # Default
                 if "phone" in found_key: v_type = "phone_detected"
                 elif "electronic" in found_key: v_type = "suspicious_object" # Earbuds map to object or phone? 
                 elif "book" in found_key: v_type = "book_detected"
                 elif "people" in found_key: v_type = "multiple_faces"
                 elif "eyes" in found_key: v_type = "gaze_deviation"
                 elif "head" in found_key: v_type = "head_orientation_issue"
                 elif "missing" in found_key: v_type = "candidate_absent"
                 
                 # Special handling for Earbuds -> Critical
                 if found_key == "unauthorized electronic":
                     points = 100 # High severity for earbuds
                     level = 5
                     v_type = "phone_detected" # Map to phone for dashboard icon compatibility
                 else:
                     points = self.VIOLATION_POINTS.get(v_type, 100)
                     level = 5 if points >= 100 else 4
                 
                 label = "CRITICAL" if level == 5 else "HIGH"
                 
                 violations.append({
                    "type": v_type, 
                    "level": level, 
                    "message": f"{label} (AI Verified): {details['reason']}",
                    "requires_review": True
                 })
                 score_increase += points
                 is_critical_frame = (level == 5)
                 critical_types_this_frame.append(v_type)
            else:
                 # Trigger verification if not already pending
                 if v_key not in self.pending_verifications:
                     # Check if we already verified the promoted key too
                     is_promoted_pending = False
                     if "phone" in v_key and "unauthorized electronic" in self.pending_verifications:
                         is_promoted_pending = True
                         
                     if not is_promoted_pending:
                         if frame is not None:
                              self.pending_verifications[v_key] = now
                              self.verifier.verify_frame_async(
                                  frame, 
                                  target, 
                                  self._callback_verified, 
                                  self._callback_rejected
                              )
                          
                 # While PENDING, we don't add high points, only a small flag
                 violations.append({
                    "type": f"verifying_{v_key}", 
                    "level": 2, 
                    "message": f"AI verify: {target}...",
                    "requires_review": False
                 })
                 # Small temporary bump to show activity in dashboard
                 score_increase += 2  # Changed from 20 (0-100 scale) 

        # ---------------------------
        # STANDARD CHECKS
        # ---------------------------

        # REMOVED: Immediate multiple_faces handling (Now moved to Gemini Hybrid logic above)
        # if detection_data.get("multiple_faces"):
        #     ...


        # if detection_data.get("candidate_absent"):
        #     points = self.VIOLATION_POINTS["candidate_absent"]
        #     violations.append({
        #         "type": "absence", 
        #         "level": 4, 
        #         "message": "HIGH: Candidate absent (>20s)", 
        #         "requires_review": False 
        #     })
        #     score_increase += points

        if detection_data.get("audio_spike"):
            points = self.VIOLATION_POINTS["audio_spike"]
            violations.append({
                "type": "audio_spike", 
                "level": 3, 
                "message": "Sustained speech detected",
                "requires_review": False
            })
            score_increase += points

        if detection_data.get("focus_loss"):
            if now - self.last_focus_loss_time > 3.0:
                points = self.VIOLATION_POINTS["focus_loss"]
                violations.append({
                    "type": "focus_loss", 
                    "level": 2, 
                    "message": "Window focus lost",
                    "requires_review": False
                })
                score_increase += points
                self.last_focus_loss_time = now

        if detection_data.get("candidate_absent"):
            points = self.VIOLATION_POINTS["candidate_absent"]
            violations.append({
                "type": "candidate_absent", 
                "level": 4, 
                "message": "HIGH: Candidate absent (>20s)", 
                "requires_review": False 
            })
            score_increase += points

        if detection_data.get("gaze_deviation") or detection_data.get("head_orientation_issue"):
            # Prioritize head label for clarity (Looking LEFT vs Gaze LEFT)
            direction = detection_data.get("head_label", detection_data.get("gaze_label", "AWAY"))
            msg = f"HIGH: {direction} (>10s)"
            points = self.VIOLATION_POINTS["gaze_deviation"]
            violations.append({
                "type": direction.replace(":", ""), 
                "level": 4, 
                "message": msg,
                "requires_review": True
            })
            score_increase += points

        if detection_data.get("excessive_movement"):
            points = self.VIOLATION_POINTS["excessive_movement"]
            violations.append({
                "type": "excessive_movement", 
                "level": 1, 
                "message": "Excessive movement",
                "requires_review": False
            })
            score_increase += points

        # Update Risk Scores
        if score_increase > 0:
             self.risk_score = max(self.risk_score, score_increase)
        
        self.session_risk_score = max(self.session_risk_score, self.risk_score)

        if is_critical_frame:
            for vtype in critical_types_this_frame:
                self.active_critical_violations[vtype] = now
        
        if len(violations) > 0:
            self.violation_history.extend([{
                **v, 
                "timestamp": now,
                "risk_at_time": self.risk_score
            } for v in violations])

        # Determine risk bands
        def get_band(score):
            if score >= self.RISK_BANDS["critical"]: return "critical"
            if score >= self.RISK_BANDS["high"]: return "high"
            if score >= self.RISK_BANDS["medium"]: return "medium"
            if score >= self.RISK_BANDS["low"]: return "low"
            return "clear"

        current_band = get_band(self.risk_score)
        session_band = get_band(self.session_risk_score)

        # Notification cooldowns
        can_notify_warn = (self.last_warn_notification is None or now - self.last_warn_notification > 60)
        can_notify_pause = (self.last_pause_notification is None or now - self.last_pause_notification > 60)

        # Suggestions
        suggest_review = (session_band == "critical")
        suggest_warn = (current_band == "medium" or current_band == "high")
        suggest_pause = (current_band == "high")
        suggest_terminate = (current_band == "critical")
        
        # NEVER auto-terminate in human-in-loop mode
        suggest_terminate = False

        # Evidence capture
        trigger_evidence = (current_band in ["high", "critical"]) or is_critical_frame
        
        # SEQUENTIAL Logic: Only capture if AI explicitly confirmed OR if it's a non-AI violation
        # (Behavioral violations like Gaze/Head/Absence should trigger immediately)
        confirmed_reason = None
        evidence_start_timestamp = None
        
        AI_TYPES = ["phone_detected", "book_detected", "multiple_faces", "suspicious_object"]
        has_non_ai_high_violation = any(v.get("level", 0) >= 4 and v["type"] not in AI_TYPES for v in violations)

        if self.trigger_confirmed_evidence:
            trigger_evidence = True
            confirmed_reason = self.trigger_confirmed_evidence["reason"]
            evidence_start_timestamp = self.trigger_confirmed_evidence["start_time"]
            # We will reset this ONLY if we actually trigger the evidence successfully (cooldown check)
        elif has_non_ai_high_violation:
            # Allow Gaze/Head/Absence to trigger evidence without AI
            trigger_evidence = True
            evidence_start_timestamp = now # Use current time for behavioral start
        else:
            # Block automatic evidence capture for High/Critical if they only contain unconfirmed AI types
            trigger_evidence = False 

        if trigger_evidence and (
            self.last_evidence_time is None or 
            now - self.last_evidence_time > CONFIG['evidence']['cooldown_seconds']
        ):
            self.last_evidence_time = now
            # Successful trigger! Now we can clear the pending confirmation
            self.trigger_confirmed_evidence = None
        else:
            trigger_evidence = False
            # If cooldown blocked it, we keep self.trigger_confirmed_evidence for the NEXT frame
            # until a clip can be saved.

        if suggest_warn and can_notify_warn: self.last_warn_notification = now
        if suggest_pause and can_notify_pause: self.last_pause_notification = now

        import re
        reason_parts = [v["type"] for v in violations if v.get("level", 0) >= 4]
        
        if confirmed_reason:
            # For filenames, use the violation type (short) rather than the long AI description
            main_type = reason_parts[0] if reason_parts else "AI_VERFIED"
            final_reason = f"AI_{main_type}"
        else:
            final_reason = "+".join(reason_parts) if reason_parts else ("critical" if is_critical_frame else "high_risk")

        # Sanitize and strictly truncate to 30 chars for Windows stability
        final_reason = re.sub(r'[^\w\-_]', '', final_reason)[:30]

        if len(violations) > 0:
            reason_str = ", ".join([v["type"] for v in violations])
            log_message(
                f"⚠️  Violations={len(violations)} [{reason_str}] | "
                f"Risk: {self.risk_score:.1f} ({current_band.upper()}) | "
                f"Session: {self.session_risk_score:.1f} ({session_band.upper()}) | "
                f"Review={'YES' if suggest_review else 'no'}"
            )

        # Prepare pending summary for Dashboard
        pending_list = []
        for vkey, start_time in self.pending_verifications.items():
            duration = int(now - start_time)
            pending_list.append(f"{vkey} ({duration}s)")

        return {
            "current_risk": self.risk_score,
            "risk_level": current_band,
            "session_risk": self.session_risk_score,
            "session_band": session_band,
            "violations": violations,
            "pending_ai": pending_list, # EXPOSE TO DASHBOARD
            "suggest_warn": suggest_warn and can_notify_warn,
            "suggest_pause": suggest_pause and can_notify_pause,
            "suggest_review": suggest_review,
            "suggest_terminate": suggest_terminate, 
            "trigger_alert": current_band in ["medium", "high", "critical"],
            "trigger_evidence": trigger_evidence,
            "evidence_reason": final_reason if trigger_evidence else None,
            "evidence_timestamp": evidence_start_timestamp, # PASS THE START TIME
            "active_critical_count": len(self.active_critical_violations),
        }

    def reset_score(self):
        """Reset for new exam session"""
        self.risk_score = 0
        self.active_critical_violations = {}
        self.violation_history = []

# ========================
# Evidence Capture
# ========================
class EvidenceCapture:
    def __init__(self, buffer, config):
        self.buffer = buffer
        self.encryptor = Encryptor()
        self.config = config
        self.evidence_dir = "evidence"
        os.makedirs(self.evidence_dir, exist_ok=True)

    def capture_and_upload(self, reason="high_risk", target_timestamp=None):
        frames, timestamps = self.buffer.get_clip(
            duration_seconds=self.config['evidence']['clip_duration_seconds'],
            target_timestamp=target_timestamp
        )
        if len(frames) < 5:
            log_message(f"⚠️  Evidence skipped: Only {len(frames)} frames available in buffer.")
            return

        # Calculate actual FPS for the VideoWriter
        actual_fps = 10 # Default
        if len(timestamps) > 1:
            total_time = timestamps[-1] - timestamps[0]
            if total_time > 0:
                actual_fps = len(timestamps) / total_time
        
        # Cap FPS between 1 and 15
        actual_fps = max(1.0, min(15.0, actual_fps))

        ts = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"evidence_{ts}_{reason}.mp4"
        file_path = os.path.join(self.evidence_dir, file_name)

        height, width, _ = frames[0].shape
        
        # Codec Fallback logic for Windows
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, actual_fps, (width, height))
        
        if not out.isOpened():
            # Fallback to AVI if MP4 fails locally
            file_path = file_path.replace(".mp4", ".avi")
            file_name = file_name.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(file_path, fourcc, actual_fps, (width, height))

        if not out.isOpened():
             log_message(f"❌ CRITICAL: Could not open VideoWriter for {reason}")
             return

        for frame in frames:
            out.write(frame)
        out.release()

        try:
            with open(file_path, 'rb') as f:
                encrypted = self.encryptor.encrypt(f.read())

            global evidence_log, current_risk_for_dashboard, current_risk_band_for_dashboard
            evidence_log.append({
                "time": time.strftime("%H:%M:%S"),
                "risk": current_risk_for_dashboard,
                "risk_band": current_risk_band_for_dashboard,
                "reason": reason,
                "file": file_name
            })

            log_message(f"📹 Evidence saved: {file_name} ({len(encrypted)} bytes)")
        except Exception as e:
            log_message(f"❌ LOGGING ERROR: Failed to process/save evidence file: {e}")
            # Ensure the file is at least released even if logging fails
            if 'out' in locals(): out.release()

# ========================
# Mock send event
# ========================
def send_event(event_data):
    log_message(f"Sending event: {event_data}")

# Global shared state for Dashboard
evidence_log = []
critical_flags = []
recent_log_lines = []
last_action = "INITIALIZING"
current_risk_for_dashboard = 0.0
current_risk_band_for_dashboard = "clear"
session_risk_for_dashboard = 0.0
session_band_for_dashboard = "clear"
engine_result_cache = {} # NEW: Global cache for RuleEngine results

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <title>AI Proctoring Dashboard [Human Review Mode]</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; background:#f3f4f6; }
    h1 { margin-bottom: 5px; color: #1f2937; }
    .subtitle { color: #6b7280; font-size: 14px; margin-bottom: 20px; }
    .mode-badge { background: #3b82f6; color: white; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    #summary { margin-bottom: 20px; background: #fff; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .label { font-weight: bold; color: #374151; }
    #logs, #evidence {
      background:#ffffff; padding:15px; border-radius:6px;
      max-height:350px; overflow-y:auto; margin-bottom:20px;
      font-size: 13px; line-height: 1.6;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .log-line { padding: 6px; margin: 2px 0; border-radius: 3px; }
    .log-line.clear { background: #f0fdf4; color:#166534; }
    .log-line.low { background: #eff6ff; color:#1e40af; }
    .log-line.medium { background: #fef3c7; color:#92400e; }
    .log-line.high { background: #fee2e2; color:#991b1b; font-weight:500; }
    .log-line.critical { background: #fecaca; color:#7f1d1d; font-weight:bold; border-left: 4px solid #dc2626; }
    .badge { padding:3px 8px; border-radius:4px; font-size:11px; color:#fff; font-weight:600; margin-left: 8px; }
    .badge.clear { background:#10b981; }
    .badge.low { background:#3b82f6; }
    .badge.medium { background:#f59e0b; }
    .badge.high { background:#ef4444; }
    .badge.critical { background:#dc2626; }
    .badge.action { background:#8b5cf6; }
    .badge.review { background:#ec4899; animation: pulse 2s infinite; }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th { background:#f9fafb; font-weight:600; color:#374151; }
    th, td { border-bottom:1px solid #e5e7eb; padding:8px 6px; text-align:left; }
    h3 { color: #1f2937; margin-top: 20px; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 11px; }
  </style>
</head>
<body>
  <h1>🎓 AI Proctoring Monitor</h1>
  <div class="subtitle">
    <span class="mode-badge">HUMAN REVIEW MODE</span>
    AI Detects → Human Decides | No Auto-Termination
  </div>
  
    <span class="label">Current Risk:</span> <span id="risk_score">0</span>
    (<span id="risk_band">clear</span>)
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span class="label">Session Max:</span> <span id="session_risk">0</span>
    (<span id="session_band">clear</span>)
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span class="label">Active Critical:</span> <span id="critical_count">0</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span class="label">AI Verifying:</span> <span id="ai_status" style="color:#d97706; font-weight:bold;">NONE</span>
  </div>

  <h3>📊 Live Event Stream (AI Detection Log)</h3>
  <div id="logs"></div>

  <h3>📹 Evidence Recordings (For Human Review)</h3>
  <div id="evidence">
    <table id="evidence_table">
      <thead>
        <tr><th>Time</th><th>Risk</th><th>Band</th><th>Reason</th><th>File</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
<script>
function updateDashboard() {
  fetch('/logs').then(r => r.json()).then(data => {
    document.getElementById('risk_score').innerText = data.current_risk.toFixed(1);
    document.getElementById('risk_band').innerText = data.risk_band;
    document.getElementById('session_risk').innerText = data.session_risk.toFixed(1);
    document.getElementById('session_band').innerText = data.session_band;
    
    document.getElementById('critical_count').innerText = data.critical_count || 0;
    
    let aiStatus = document.getElementById('ai_status');
    if (data.pending_ai && data.pending_ai.length > 0) {
      aiStatus.innerText = data.pending_ai.join(', ');
      aiStatus.style.animation = 'pulse 1.5s infinite';
    } else {
      aiStatus.innerText = 'NONE';
      aiStatus.style.animation = 'none';
    }

    let logsDiv = document.getElementById('logs');
    logsDiv.innerHTML = '';
    data.logs.slice().reverse().slice(0, 50).forEach(function(item) {
      let div = document.createElement('div');
      div.className = 'log-line ' + (item.risk_band || 'clear');
      let badge = '<span class="badge ' + (item.risk_band || 'clear') + '">' +
                  (item.risk_band || 'clear').toUpperCase() + '</span>';
      let txt = '[' + item.time + '] ' + badge + ' ' + item.message;
      if (item.action) {
        txt += ' <span class="badge action">' + item.action + '</span>';
      }
      if (item.review) {
        txt += ' <span class="badge review">NEEDS REVIEW</span>';
      }
      div.innerHTML = txt;
      logsDiv.appendChild(div);
    });

    let tbody = document.querySelector('#evidence_table tbody');
    tbody.innerHTML = '';
    data.evidence.slice().reverse().slice(0, 20).forEach(function(ev) {
      let tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + ev.time + '</td>' +
        '<td>' + ev.risk.toFixed(1) + '</td>' +
        '<td><span class="badge ' + ev.risk_band + '">' + ev.risk_band.toUpperCase() + '</span></td>' +
        '<td>' + ev.reason + '</td>' +
        '<td><code>' + ev.file + '</code></td>';
      tbody.appendChild(tr);
    });
  });
}
setInterval(updateDashboard, 2000);
updateDashboard();
</script>
</body>
</html>
""")
@app.route('/logs')
def get_logs():
    global current_risk_for_dashboard, current_risk_band_for_dashboard
    global session_risk_for_dashboard, session_band_for_dashboard
    return {
        "current_risk": current_risk_for_dashboard,
        "risk_band": current_risk_band_for_dashboard,
        "session_risk": session_risk_for_dashboard,
        "session_band": session_band_for_dashboard,
        "last_action": last_action,
        "critical_count": len(critical_flags),
        "logs": recent_log_lines[-100:],
        "evidence": evidence_log[-50:],
        "pending_ai": engine_result_cache.get("pending_ai", []) # We need a way to pass result here
    }
#========================
#Main monitoring loop (FIXED)
#========================
def start_monitoring():
    try:
        buffer = RollingBuffer(
            max_seconds=CONFIG['detection']['buffer_max_history_seconds'],
            fps=CONFIG['detection']['frame_interval_fps']
        )
        detector = VisualDetector()
        engine = RuleEngine(CONFIG)
        system_monitor = SystemMonitor()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_message("ERROR: Cannot access webcam")
            return
        
    except Exception as e:
        log_message(f"CRITICAL INIT ERROR: {e}")
        traceback.print_exc()
        return

    last_frame_time = 0
    frame_interval = 1.0 / CONFIG['detection']['frame_interval_fps']

    audio_data = {"audio_spike": False, "speech_probability": 0.0}

    def audio_callback(d):
        nonlocal audio_data
        audio_data = d

    print(">>> [THREAD] Initializing AudioMonitor (Silero VAD)...", flush=True)
    audio_monitor = AudioMonitor(audio_callback)
    audio_monitor.start()

    log_message("✅ Monitoring started - HUMAN REVIEW MODE")
    log_message(f"📊 Config: {CONFIG['detection']['frame_interval_fps']} FPS, "
                f"Gaze: {CONFIG['thresholds']['gaze_deviation_seconds']}s, "
                f"Absence: {CONFIG['thresholds']['absence_seconds']}s")
    log_message("🧑‍💼 AI generates flags → Human proctor decides")

    global critical_flags

    try:
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count % 50 == 0: # Every ~5 seconds
                print(f"DEBUG: Monitoring Loop Alive ({loop_count})")

            now = time.time()
            if now - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            last_frame_time = now

            ret, frame = cap.read()
            if not ret:
                print("DEBUG: Frame capture failed (Camera busy?)")
                continue

            try:
                buffer.add_frame(frame)
                visual = detector.process_frame(frame)
                focus_loss = system_monitor.check_focus_loss()
                
                # HEARTBEAT: Log occasionally to show loop is running
                if loop_count % 100 == 0:
                    log_message(f"DEBUG: Processing Frame {loop_count} (Risk: {engine.risk_score:.1f})")

                detection = {
                    **visual,
                    "audio_spike": audio_data.get("audio_spike", False),
                    "focus_loss": focus_loss
                }

                result = engine.process_detection(detection, frame)
            
            except Exception as inner_e:
                print(f"CRITICAL ERROR in Loop: {inner_e}")
                traceback.print_exc()
                continue

            global current_risk_for_dashboard, current_risk_band_for_dashboard
            global session_risk_for_dashboard, session_band_for_dashboard
            global last_action, recent_log_lines, engine_result_cache

            current_risk_for_dashboard = result["current_risk"]
            current_risk_band_for_dashboard = result["risk_level"]
            session_risk_for_dashboard = result["session_risk"]
            session_band_for_dashboard = result["session_band"]
            engine_result_cache = result # Store full result for dashboard

            # FIXED: Only log when there are actual violations
            if len(result.get("violations", [])) > 0:
                violation_msgs = [v["message"] for v in result["violations"]]
                msg = "; ".join(violation_msgs)
            
                action_label = None
                needs_review = result.get("suggest_review", False)
            
                if needs_review:
                    action_label = "🔍 REVIEW"
                    # Add to critical flags list
                    critical_flags.append({
                        "time": time.strftime("%H:%M:%S"),
                        "violations": result["violations"],
                        "risk": result["current_risk"]
                    })
                elif result["suggest_terminate"]:
                    action_label = "❌ TERMINATE?"
                elif result["suggest_pause"]:
                    action_label = "⏸️ PAUSE"
                elif result["suggest_warn"]:
                    action_label = "⚠️ WARN"

                if action_label:
                    last_action = action_label

                recent_log_lines.append({
                    "time": time.strftime("%H:%M:%S"),
                    "message": msg,
                    "risk_band": result["risk_level"],
                    "action": action_label,
                    "review": needs_review
                })

            # EVIDENCE CAPTURE (Moved outside to ensure AI triggers aren't lost if object is put away)
            if result.get("trigger_evidence"):
                reason = result.get("evidence_reason", "high_risk")
                start_ts = result.get("evidence_timestamp")
                log_message(f"📹 Evidence Triggered: {reason} (Detection started at {start_ts})")
                EvidenceCapture(buffer, CONFIG).capture_and_upload(reason=reason, target_timestamp=start_ts)

            # Display Risk Info (Top)
            color = (0, 255, 0) if result['risk_level'] in ['clear', 'low'] else \
                    (0, 165, 255) if result['risk_level'] == 'medium' else \
                    (0, 0, 255)
        
            cv2.putText(
                frame, 
                f"Risk: {result['current_risk']:.1f} ({result['risk_level'].upper()})",
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                color,
                2
            )
        
            if result.get("active_critical_count", 0) > 0:
                cv2.putText(
                    frame,
                    f"CRITICAL ACTIVE: {result['active_critical_count']}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # AI Verifying Status Indicator (Top Right)
            pending_items = result.get("pending_ai", [])
            if pending_items:
                # Flashing effect every 5 frames
                if (loop_count // 5) % 2 == 0:
                    status_text = f"AI VERIFYING: {', '.join(pending_items)}"
                    cv2.putText(
                        frame,
                        status_text,
                        (w - 350, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2
                    )

            # NEW: Enhanced Camera Preview (Movement Stats - Bottom)
            # Draw a dark background for the text for better readability
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
            
            # Yaw/Pitch
            cv2.putText(
                frame,
                f"Degrees: Yaw {visual.get('yaw', 0):.1f}, Pitch {visual.get('pitch', 0):.1f}",
                (10, h-35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            # Labels
            cv2.putText(
                frame,
                f"Labels: {visual.get('head_label', '')} | {visual.get('gaze_label', '')}",
                (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
        
            cv2.imshow("Proctoring [Human Review Mode]", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        audio_monitor.stop()
        cap.release()
        cv2.destroyAllWindows()
        log_message("✅ Monitoring stopped")
#========================
#Entry point
#========================
if __name__ == "__main__":
    print("=" * 60)
    print("AI PROCTORING SYSTEM - Human-in-the-Loop Mode")
    print("=" * 60)
    print("✅ AI generates detection flags")
    print("✅ Human proctor makes final decisions")
    print("✅ NO auto-termination")
    print("=" * 60)
    print(f"Frame Rate: {CONFIG['detection']['frame_interval_fps']} FPS")
    print(f"Gaze Threshold: {CONFIG['thresholds']['gaze_deviation_seconds']}s") 
    print(f"Absence Threshold: {CONFIG['thresholds']['absence_seconds']}s")
    print(f"Audio Threshold: {CONFIG['thresholds']['audio_sustained_seconds']}s")
    print("=" * 60)
    print("Dashboard: http://localhost:8000")
    print("=" * 60)
    print("Starting Monitor Thread...", flush=True)
    monitor_thread = threading.Thread(target=start_monitoring, daemon=True)
    monitor_thread.start()

    app.run(host="0.0.0.0", port=8000, debug=False)
