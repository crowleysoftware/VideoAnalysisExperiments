import onnxruntime as ort

session = ort.InferenceSession("C:/Users/Administrator/Downloads/discvisioon.ONNX/model.onnx")

import cv2
import numpy as np

def preprocess(image_path):
    image = cv2.imread(image_path)
    input_size = (416, 416)  # Example size, adjust based on your model
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(image_transposed, axis=0).astype(np.float32)
    return input_tensor

input_tensor = preprocess("C:/Users/Administrator/OneDrive - Newup/Pictures/Screenshots/top.png")

inputs = {session.get_inputs()[0].name: input_tensor}
outputs = session.run(None, inputs)

def postprocess(outputs, image):
    boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Class {class_id}: {score:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

image = cv2.imread('input_image.jpg')
result_image = postprocess(outputs, image)
cv2.imshow('Detections', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
