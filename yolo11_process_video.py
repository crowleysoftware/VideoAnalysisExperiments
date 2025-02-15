import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

video_path = "C:/Users/Administrator/OneDrive - Newup (1)/Recordings/PXL_20250215_190728005.mp4"

model = YOLO("best.pt")

results = model(video_path)
output_list = []

frame_count = 0
last_was_empty = False

for result in results:
    frame_count += 1
    json_str = result.to_json()
    rdict = json.loads(json_str)
    
    if len(rdict) > 0 and rdict[0]["confidence"] < 0.85:
        continue
    
    if rdict == [] and last_was_empty:
        continue
    
    # if rdict is empty, set confidence to 0 and class to -1
    if rdict == []:
        last_was_empty = True
        rdict.append({
            "confidence": 0,
            "class": -1
        })
    else:
        last_was_empty = False
        if rdict[0]["confidence"] < 0.85:
            continue
    
    output_dict = {
        "frame_count": frame_count,
        "class": rdict[0]["class"],
        "confidence": rdict[0]["confidence"]
    }

    output_list.append(output_dict)
    
    result.save_crop("C:/repos/yolo_experiment/frames", f"crop_{frame_count}.jpg")
    
# loop over output_list
temp_confidence = 0
temp_index = 0
chosen = []
for item in output_list:
    if item["class"] == 0:
        if item["confidence"] > temp_confidence:
            temp_confidence = item["confidence"]    
            temp_index = output_list.index(item)
    else:
        temp_confidence = 0
        frame = output_list[temp_index]["frame_count"]
        results[frame].save_crop("C:/repos/yolo_experiment/chosen_frames", f"crop_{frame}.jpg")
        chosen.append(output_list[temp_index])

# write to file
with open("video_detection_log.txt", "a") as f:
    f.write(json.dumps(output_list) + "\n")
    
with open("chosen.txt", "a") as f:
    f.write(json.dumps(chosen) + "\n")

