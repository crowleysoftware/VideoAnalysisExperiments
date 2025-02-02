import cv2
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolo11n.pt")

# Display model information (o

# Configure logging
logging.basicConfig(filename='detection_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load video
video_path = "C:/Users/Administrator/OneDrive - Newup (1)/Recordings/PXL_20250201_171030199.mp4"
cap = cv2.VideoCapture(video_path)

# Get video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)


    # Render results
   # results.render()

    # Save frame with detection boxes (optional)
    cv2.imwrite(f'frames/frame_{frame_count:04d}.jpg', results.ims[0])

    frame_count += 1

    # Write the frame with detection boxes
    out.write(results.ims[0])

    # Display the frame (optional)
    cv2.imshow('YOLOv5 Detection', frame) #results.ims[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
