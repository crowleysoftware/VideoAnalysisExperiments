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

class DetectionResult:
    def __init__(self, cls, confidence, result_number, color, aspect_ratio, class_name):
        self.cls = cls
        self.confidence = confidence
        self.result_number = result_number
        self.color = color
        self.aspect_ratio = aspect_ratio
        self.class_name = class_name

    def __repr__(self):
        return f"DetectionResult(class={self.cls}, confidence={self.confidence}, result_number={self.result_number}, color={self.color}, class_name={self.class_name})"

    def to_dict(self):
        return {
            "class": self.cls,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "result_number": self.result_number,
            "color": self.color,
            "aspect_ratio": self.aspect_ratio
        }
        
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

def get_the_color(convert_rgb_to_names, get_predominant_color, roi):
    predominant_color = get_predominant_color(roi)
    predominant_color_tuple = tuple(map(int, predominant_color))
    mycolor = convert_rgb_to_names(predominant_color_tuple)
    return predominant_color,mycolor

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolo11n.pt")

# Configure logging
logging.basicConfig(filename='detection_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load video
video_path = "C:/Users/Administrator/OneDrive - Newup (1)/Recordings/PXL_20250202_200826524.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
frames_to_skip = 5
skipped_frames = 0
result_nbr = 0

# List to collect detection results
detection_results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if skipped_frames < frames_to_skip:
        skipped_frames += 1
        continue
    else:
        skipped_frames = 0        
    
    # object detection happens here    
    results = model(frame)
       
    for result in results:
        result_nbr += 1
        json_str = result.to_json()
        logging.info(json_str)
        
        # Parse JSON string to a dictionary
        result_dict = json.loads(json_str)
        
        if result_dict == []:
            continue
        
        for item in result_dict:            
            name = item["name"]                
            cls = item["class"]
            confidence = item["confidence"]
            box = item["box"]
            y1, x1, y2, x2 = int(box["y1"]), int(box["x1"]), int(box["y2"]), int(box["x2"])
            
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = round(width / height, 1)
            
            #log aspect ratio
            logging.info(f'Frame {frame_count}: Aspect Ratio {aspect_ratio}')
            
            if not (0.8 <= aspect_ratio <= 1.2):  # Adjust the tolerance as needed
                continue

            if confidence < 0.5:
                continue
            
            # Extract the ROI using the bounding box coordinates
            roi = frame[y1:y2, x1:x2]
            
            # Get the predominant color in the ROI
            predominant_color, mycolor = get_the_color(convert_rgb_to_names, get_predominant_color, roi)

            logging.info(f'Frame {frame_count}, RGB {predominant_color}, Color {mycolor}')

            detection_results.append(DetectionResult(cls, confidence, result_nbr, mycolor, aspect_ratio, name))            

        # Check if any item in result_dict has name equal to "person"
        if any(item["name"] == "person" for item in result_dict):
            logging.info(f"Skipping saving frame {frame_count} as it contains a person")
        else:
            result.save_crop("C:/repos/yolo_experiment/frames", f"frame_{result_nbr}.jpg")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Convert detection results to a list of dictionaries
detection_results_dicts = [dr.to_dict() for dr in detection_results]

#todo: decide which results to save. Pair up top and bottom images
#find the result that is half way between the first frame and the first frame that detected a person
position = 0
last_person_frame = 0

for dr in detection_results:
    if dr.class_name == "person":
        person_frame = position
        img_selection = last_person_frame +  ((position - last_person_frame) // 2)
        logging.info(f"Person frame: {person_frame}, Last person frame: {last_person_frame}, Position: {position}, Image selection: {img_selection}, Frame: {detection_results[position].result_number}")
        last_person_frame = position
        
        #now keep going until we find non person frame
        
            
    position += 1

# Serialize the list of dictionaries to JSON and write it to a file
with open('detection_results.json', 'w') as json_file:
    json.dump(detection_results_dicts, json_file, indent=4)