# SVRACS  
# Smart Vehicle Recognition and Access Control System using Nvidia Jetson Nano

This project implements a smart and automated solution for managing vehicle access to secure areas. It uses computer vision and machine learning to streamline the entry and exit process, eliminating the need for manual vehicle verification. The system handles everything from recognizing vehicles and reading license plates to generating a digital pass in the form of a QR code, which is then used for exit verification.

# Key Features
1. Automated Vehicle Recognition: Uses YOLOv5s to detect and classify vehicle types.
2. License Plate Detection and Recognition: Leverages YOLOv5s for license plate detection and Tesseract OCR to read the plate numbers.
3. Color Identification: Uses ResNet50 to determine the color of the vehicle for enhanced verification.
4. QR Code for Digital Pass: Vehicle information is encrypted and encoded in a QR code, which is used as a digital pass.
5. Streamlined Exit Process: The QR code is scanned at the exit to quickly verify vehicle details, making the process more efficient and secure.

# Technologies Used
1. NVIDIA Jetson Nano - For model deploymenr
2. USB Camera - For live video capture
3. YOLOv5s: For vehicle detection and license plate identification.
4. ResNet50: For vehicle color recognition.
5. Tesseract OCR: For reading license plate characters.
6. Streamlit: To create a user-friendly QR code scanning interface.
7. QR Code Generation: For creating a secure, digital pass.
8. Thermal Printer: To print the QR codes for the drivers.

# How to Use
Image Inference:
To run inference on a static image, use the img.py or img1.py script.
These scripts will detect the vehicle, recognize the license plate, and classify the vehicle's color.  
              run:  python3 img.py --image <path_to_image>

Video Inference:
For video files, use the vid.py or vid1.py script to perform inference on a recorded video.
This will allow the system to process the entire video, detecting vehicles and license plates frame by frame.  
             run:  python3 vid.py --video <path_to_video>
             
Live Stream Inference:
To run inference on a live stream from a USB camera, use the live.py or live1.py script.
This script processes live video input to detect vehicles, recognize license plates, and classify the vehicle's color in real time.  
             run:  python3 live.py
# Model Files: You can download the model files using this link -  
https://drive.google.com/drive/folders/1vosXcIQgdshpWzcyOAymPLzlTqLlsD0D?usp=sharing  
The models/ folder contains the ResNet50 model used for vehicle color classification.  
The build/ and build1/ folders contain the YOLOv5s TensorRT engine used for license plate and vehicle recognition respectively.

# References
https://www.youtube.com/watch?v=ErWC3nBuV6k  
https://github.com/dusty-nv/jetson-inference

