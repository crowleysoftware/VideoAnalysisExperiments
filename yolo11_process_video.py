
import numpy as np
from pathlib import Path
from ultralytics import YOLO

video_path = "C:/Users/Administrator/OneDrive - Newup (1)/Recordings/PXL_20250202_200826524.mp4"

model = YOLO("yolo11n.pt")

results = model(video_path)

for result in results:
    json_str = result.to_json()
    # write to file
    with open("video_detection_log.txt", "a") as f:
        f.write(json_str + "\n")

