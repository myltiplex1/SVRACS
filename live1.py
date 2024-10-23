import cv2
from yoloDet import YoloTRT
import yoloDet1
from jetson_inference import imageNet
from jetson_utils import cudaFromNumpy
import os
import numpy as np
import qrcode
import time

# Initialize the YOLO models
plate_model = YoloTRT(library="build/libmyplugins.so", engine="build/best.engine", conf=0.5, yolo_ver="v5")
vehicle_model = yoloDet1.YoloTRT(library="build1/libmyplugins.so", engine="build1/yolov5s.engine", conf=0.5, yolo_ver="v5")

# Initialize the imageNet model for color classification
color_net = imageNet(model="models/color_v12/resnet50.onnx", labels="data/color/labels.txt", 
                     input_blob="input_0", output_blob="output_0")

def classify_color(image):
    # Convert image to CUDA format
    image_cuda = cudaFromNumpy(image)

    # Perform classification
    class_id, _ = color_net.Classify(image_cuda)
    predicted_label = color_net.GetClassLabel(class_id)
    return predicted_label

# Define GStreamer pipeline for video capture
gst_pipeline = ('v4l2src device=/dev/video0 ! ' +
                'image/jpeg,width=640,height=480,framerate=30/1 ! ' +
                'jpegdec ! videoconvert ! appsink')

# Open video capture device
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
cv2.namedWindow('Vehicle and Plate Detection', cv2.WINDOW_NORMAL)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

# Initialize QR code window ID and QR image path
qr_window_id = None
qr_image_path = None

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame using the YOLO model for vehicle detection
    vehicle_detections, _ = vehicle_model.Inference(frame)

    # Create a dictionary to store detected vehicle types and their plate numbers
    vehicle_info = {}

    # Extract vehicle types from detections and add them to the dictionary
    for detection in vehicle_detections:
        vehicle_type = detection["class"]
        if vehicle_type in ["car", "motorcycle", "bus", "truck"]:
            vehicle_bbox = detection["box"]
            x1, y1, x2, y2 = map(int, vehicle_bbox)  # Ensure bounding box coordinates are integers
            vehicle_img = frame[y1:y2, x1:x2]
            
            # Introduce a delay of 3 seconds (adjust as necessary)
            time.sleep(3)
            
            # Extract color features and make prediction using imageNet
            color_prediction = classify_color(vehicle_img)
            
            # Store vehicle information
            vehicle_info[vehicle_type] = {"type": vehicle_type, "plate_number": None, "color": color_prediction}

    # Perform inference on the frame using the YOLO model for license plate detection
    plate_detections, plate_numbers, _ = plate_model.Inference(frame)

    # Update vehicle information with plate numbers
    for plate_number in plate_numbers:
        for info in vehicle_info.values():
            if info["plate_number"] is None:
                info["plate_number"] = plate_number
                break

    # Close QR code window if no vehicles are detected
    if not vehicle_info and qr_window_id is not None:
        cv2.destroyWindow("QR Code")
        qr_window_id = None
        qr_image_path = None

    # Generate QR code containing vehicle information if vehicle_info is not empty
    if vehicle_info:
        qr_data = "\n".join([f"Vehicle Type: {info['type']}\nPlate Number: {info['plate_number'].strip() if info['plate_number'] else 'N/A'}\nColor: {info['color']}" for info in vehicle_info.values()])
        qr = qrcode.make(qr_data)

        # Convert the QR code to a format OpenCV can handle
        qr_pil = qr.convert('RGB')
        qr_cv = np.array(qr_pil)
        qr_cv = cv2.cvtColor(qr_cv, cv2.COLOR_RGB2BGR)

        # Display QR code and save it if not saved already
        if qr_window_id is None:
            cv2.imshow("QR Code", qr_cv)
            qr_window_id = cv2.getWindowProperty("QR Code", cv2.WND_PROP_VISIBLE)

            # Save QR code image
            qr_image_path = "vehicle_info_qr.png"
            qr_pil.save(qr_image_path)
            print(f"QR code containing vehicle information saved to: {qr_image_path}")

    # Display detected vehicle types and plate numbers on the screen
    for info in vehicle_info.values():
        cv2.putText(frame, f"Vehicle Type: {info['type']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Plate Number: {info['plate_number']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Color: {info['color']}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Vehicle and Plate Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

