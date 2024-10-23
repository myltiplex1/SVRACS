import cv2
import argparse
from yoloDet import YoloTRT
import yoloDet1
from jetson_inference import imageNet
from jetson_utils import cudaFromNumpy
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

# Argument parser for video path
parser = argparse.ArgumentParser(description="Vehicle and License Plate Detection")
parser.add_argument('--video', required=True, help="Path to the video file")
args = parser.parse_args()

# Open video file
cap = cv2.VideoCapture(args.video)
cv2.namedWindow('Vehicle and Plate Detection', cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

# Initialize QR code window ID
qr_window_id = None

# Read video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame using the YOLO model for vehicle detection
    vehicle_detections, _ = vehicle_model.Inference(frame)

    # Perform inference on the frame using the YOLO model for license plate detection
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
        qr_data = "\n".join([f"Vehicle Type: {info['class']}\nPlate Number: {info['plate_number'] if info['plate_number'] else 'N/A'}\nColor: {info['color']}" for info in vehicle_info])
        qr = qrcode.make(qr_data)

        qr_pil = qr.convert('RGB')
        qr_cv = np.array(qr_pil)
        qr_cv = cv2.cvtColor(qr_cv, cv2.COLOR_RGB2BGR)

        # Close previous QR code window if it exists
        if qr_window_id is not None:
            cv2.destroyWindow("QR Code")

        # Display new QR code
        cv2.imshow("QR Code", qr_cv)
        qr_window_id = cv2.getWindowProperty("QR Code", cv2.WND_PROP_VISIBLE)

    else:
        # Close QR code window if no vehicles are detected
        if qr_window_id is not None:
            cv2.destroyWindow("QR Code")
            qr_window_id = None

    # Display vehicle info on the frame
    for i, info in enumerate(vehicle_info):
        cv2.putText(frame, f"Vehicle Type: {info['class']}", (50, 50 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Plate Number: {info['plate_number']}", (50, 100 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Color: {info['color']}", (50, 150 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Vehicle and Plate Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

