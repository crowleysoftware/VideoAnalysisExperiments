import cv2
import logging
import numpy as np
import json
import webcolors
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.spatial import KDTree
from webcolors import (
    hex_to_rgb,
)

def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = webcolors._definitions._CSS3_HEX_TO_NAMES # css3_hex_to_names
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]

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

def get_color_namex(rgb_color):
    # Convert the predefined colors and the input color to LAB color space
    lab_colors = {name: cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0] for name, rgb in colors.items()}
    lab_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2LAB)[0][0]
    
    min_distance = float('inf')
    color_name = None
    for name, lab in lab_colors.items():
        dist = distance.euclidean(lab_color, lab)
        if dist < min_distance:
            min_distance = dist
            color_name = name
    return color_name

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors._definitions._CSS21_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_color):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(rgb_color)
    except ValueError:
        closest_name = closest_color(rgb_color)
        actual_name = None
    return actual_name, closest_name

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
            y1, x1, y2, x2 = int(box["y1"]), int(box["x1"]), int(box["y2"]), int(box["x2"])
            
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height
            #log aspect ratio
            logging.info(f'Frame {frame_count}: Aspect Ratio {aspect_ratio}')
            if not (0.8 <= aspect_ratio <= 1.2):  # Adjust the tolerance as needed
                save_crop = False
                continue

            if name == 'frisbee' or name == 'clock':        
                
                if confidence < 0.5:
                    save_crop = False
                    continue
                
                save_crop = True
               
                # Extract the ROI using the bounding box coordinates
                roi = frame[y1:y2, x1:x2]
                
                # Get the predominant color in the ROI
                predominant_color = get_predominant_color(roi)
                #color_name = get_color_name(predominant_color)
                predominant_color_tuple = tuple(map(int, predominant_color))
                mycolor = convert_rgb_to_names(predominant_color_tuple)
                actual_name, color_name = get_color_name(predominant_color_tuple)
                logging.info(f'Frame {frame_count}, RGB {predominant_color}, Color {mycolor}, Actual Color {actual_name}')
            else:
                save_crop = False
        
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
