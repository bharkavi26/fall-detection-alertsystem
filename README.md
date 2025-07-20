# fall-detection-alertsystem
To address the challenges of detecting falls accurately and efficiently, this project proposes an AIpowered computer vision system capable of identifying fall incidents in real-time. The system is lightweight, cost-effective, and runs on a local machine using Python and TensorFlow Lite. The solution is divided into the following components:
Figure 1:Flow Chart of Model Implementation
1. Dataset Preparation and Model Training
A well-labelled dataset consisting of "fall" and "non-fall" images was used for training the model.
These images represented various human postures and scenarios . The dataset
was uploaded to Edge Impulse Studio, a platform for building embedded machine learning models.
Using its graphical tools, the dataset was split, pre-processed, and fed into a CNN-based model
architecture.
 Model Type: Convolutional Neural Network (CNN)
 Input Image Size: 96 × 96 pixels
 Output Classes: Fall (0) and Non-Fall (1)
After training, the model was exported as a .tflite (TensorFlow Lite) file, which is optimized for
edge devices.
2. Real-Time Fall Detection Using Webcam
The trained .tflite model was integrated into a Python script that accesses the system's webcam.
Each frame is resized, normalized, and passed to the model for inference. The system continuously
checks whether the person is in a fall or non-fall posture.
 Libraries Used: OpenCV, NumPy, TensorFlow
3.SMS Alert Integration with Twilio
To enhance safety, the system monitors the duration of the fall condition. If the same "Fall"
prediction persists continuously for more than 30 seconds, it sends an SMS alert using
the Twilio API to a pre-verified contact number.
 Condition Monitoring: Timer starts when fall is detected
 Alert Trigger: If fall persists >30 seconds
 SMS Content: Immediate notification stating a fall has occurred
This ensures timely communication to caretakers or emergency contacts, potentially reducing the
delay in getting help.
