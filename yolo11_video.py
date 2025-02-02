import cv2
import logging
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans

def get_predominant_color(image, k=1):
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to speed up processing (optional)
    resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Reshape the image to a 2D array of pixels
    pixels = resized_image.reshape((-1, 3))
    
    # Perform K-means clustering to find the predominant color
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the cluster centers (predominant colors)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Get the number of pixels in each cluster
    counts = np.bincount(kmeans.labels_)
    
    # Find the cluster with the most pixels
    predominant_color = colors[np.argmax(counts)]
    
    return predominant_color

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
       
    for result in results:

        json_str = result.to_json()
        logging.info(json_str)
        
        # Parse JSON string to a dictionary
        result_dict = json.loads(json_str)
        
        if result_dict == []:
            continue
        
        save_crop = False
        for item in result_dict:
        
            name = item["name"]
            cls = item["class"]
            confidence = item["confidence"]
            box = item["box"]

            if name == 'person' or name == 'frisbee' or name == 'clock':        
                save_crop = True
                                        
                # Extract the ROI using the bounding box coordinates
                roi = frame[int(box["y1"]):int(box["y2"]), int(box["x1"]):int(box["x2"])]
                
                # Get the predominant color in the ROI
                predominant_color = get_predominant_color(roi)
                logging.info(f'Frame {frame_count}: Predominant Color {predominant_color}')
        
        if save_crop:
            result.save_crop("C:/repos/yolo_experiment/frames")
        
    # Save frame with detection boxes (optional)
    #cv2.imwrite(f'frames/frame_{frame_count:04d}.jpg', results.ims[0])

    frame_count += 1

    # Write the frame with detection boxes
    #out.write(results.ims[0])

    # Display the frame (optional)
    #cv2.imshow('YOLOv5 Detection', frame) #results.ims[0])
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

cap.release()
out.release()
cv2.destroyAllWindows()
