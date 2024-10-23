import cv2
import argparse
from yoloDet import YoloTRT
import yoloDet1
from jetson_inference import imageNet
from jetson_utils import cudaFromNumpy
import os
import numpy as np
import qrcode

# Initialize YOLO models
plate_model = YoloTRT(library="build/libmyplugins.so", engine="build/best.engine", conf=0.5, yolo_ver="v5")
vehicle_model = yoloDet1.YoloTRT(library="build1/libmyplugins.so", engine="build1/yolov5s.engine", conf=0.5, yolo_ver="v5")

# Initialize the imageNet model for color classification
color_net = imageNet(model="models/color_v4/resnet50.onnx", labels="data/color/labels.txt",
                     input_blob="input_0", output_blob="output_0")

# Classify vehicle color
def classify_color(image):
    image_cuda = cudaFromNumpy(image)
    class_id, _ = color_net.Classify(image_cuda)
    return color_net.GetClassLabel(class_id)

# Argument parser for folder path
parser = argparse.ArgumentParser(description="Vehicle and License Plate Detection")
parser.add_argument('--folder', required=True, help="Path to the folder containing images")
args = parser.parse_args()

# Check if folder exists
if not os.path.isdir(args.folder):
    print("Error: Specified folder does not exist")
    exit()

# Loop through all images in the folder
for filename in os.listdir(args.folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Process image files only
        image_path = os.path.join(args.folder, filename)

        # Open image file
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Unable to open image file {filename}")
            continue

        # Perform inference on the image using the YOLO model for vehicle detection
        vehicle_detections, _ = vehicle_model.Inference(frame)

        # Perform inference on the image using the YOLO model for license plate detection
        plate_detections, plate_numbers, _ = plate_model.Inference(frame)

        # Create a list to store vehicle info
        vehicle_info = []

        # Extract vehicle types and check for license plates within vehicle bounding boxes
        for detection in vehicle_detections:
            vehicle_class = detection["class"]
            if vehicle_class in ["car", "motorcycle", "bus", "truck"]:
                vehicle_bbox = detection["box"]
                x1, y1, x2, y2 = map(int, vehicle_bbox)
                vehicle_img = frame[y1:y2, x1:x2]

                # Classify the color of the vehicle
                color_prediction = classify_color(vehicle_img)

                # Initialize plate number as None
                plate_number = None

                # Check for license plate overlap within the vehicle bounding box
                for plate_detection, plate_num in zip(plate_detections, plate_numbers):
                    plate_bbox = plate_detection["box"]
                    px1, py1, px2, py2 = map(int, plate_bbox)

                    if (px1 >= x1 and py1 >= y1 and px2 <= x2 and py2 <= y2):
                        plate_number = plate_num.strip()
                        break

                # Store vehicle info
                vehicle_info.append({"class": vehicle_class, "plate_number": plate_number, "color": color_prediction})

        # Generate and display QR code if vehicles are detected
        if vehicle_info:
            qr_data = "\n".join([f"Vehicle Type: {info['class']}\nPlate Number: {info['plate_number']}\nColor: {info['color']}" for info in vehicle_info])
            qr = qrcode.make(qr_data)

            qr_pil = qr.convert('RGB')
            qr_cv = np.array(qr_pil)
            qr_cv = cv2.cvtColor(qr_cv, cv2.COLOR_RGB2BGR)

            # Display QR code
            cv2.imshow(f"QR Code - {filename}", qr_cv)

        # Display vehicle info on the frame
        for i, info in enumerate(vehicle_info):
            cv2.putText(frame, f"Vehicle Type: {info['class']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 225, 225), 2)
            cv2.putText(frame, f"Vehicle Type: {info['plate_number']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 225, 225), 2)
            cv2.putText(frame, f"Color: {info['color']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 225, 225), 2)

        # Show the image with vehicle info
        cv2.imshow(f'Vehicle and Plate Detection - {filename}', frame)

        # Wait for a key press before processing the next image
        cv2.waitKey(0)
        cv2.destroyAllWindows()

