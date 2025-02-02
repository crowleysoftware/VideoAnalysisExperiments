import cv2
import torch
import logging
import numpy as np
from scipy.spatial import distance
from pathlib import Path

# Configure logging
logging.basicConfig(filename='detection_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
video_path = "C:/Users/Administrator/OneDrive - Newup (1)/Recordings/PXL_20250201_171030199.mp4"
cap = cv2.VideoCapture(video_path)

# Get video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Define a list of common colors and their RGB values
colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42)
}

def get_color_name(rgb_color):
    min_distance = float('inf')
    color_name = None
    for name, rgb in colors.items():
        dist = distance.euclidean(rgb_color, rgb)
        if dist < min_distance:
            min_distance = dist
            color_name = name
    return color_name

frame_count = 0
skip_frames = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if skip_frames < 5:
        skip_frames += 1
        continue
    else:
        skip_frames = 0
        
    results = model(frame)

    # Log detection results
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        logging.info(f'Frame {frame_count}: Class {int(cls)}, Confidence {conf:.2f}, Box {box}')

        if int(cls) == 29 or int(cls) == 74:  # Check if the detected object is a frisbee or clock
            
            if conf < 0.7:  # Check if the confidence score is below the threshold
                continue
          
            # Check if the bounding box is approximately square
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height
            #log aspect ratio
            logging.info(f'Frame {frame_count}: Aspect Ratio {aspect_ratio}')
            if not (0.8 <= aspect_ratio <= 1.2):  # Adjust the tolerance as needed
                continue
           
        # Extract the ROI using the bounding box coordinates
        roi = frame[y1:y2, x1:x2]

        # Calculate the average color of the ROI
        avg_color_per_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_color = avg_color.astype(int)
        color_name = get_color_name(avg_color)
        logging.info(f'Frame {frame_count}: Average Color {color_name}')
        
        #logging.info(f'Frame {frame_count}: Predominant Color {color_name}')
        
        # Draw the bounding box and label on the frame
        label = f'{model.names[int(cls)]} {conf:.2f} Color: {color_name}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check if there are any detections
    if len(results.xyxy[0]) > 0:
        # Render results
        results.render()

        # Save frame with detection boxes (optional)
        cv2.imwrite(f'frames/frame_{frame_count:04d}.jpg', results.ims[0])

        # Write the frame with detection boxes
        out.write(results.ims[0])

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
