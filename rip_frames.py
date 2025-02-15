import cv2
import numpy as np

from pathlib import Path
from ultralytics import YOLO

# Load video
video_path = "C:/Users/Administrator/OneDrive - Newup (1)/Recordings/latest.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # write full frame to disk
    cv2.imwrite(f"C:/repos/yolo_experiment/full_frames/frame_{frame_count}.jpg", frame)
    frame_count += 1