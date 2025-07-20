import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r'C:\Users\malai\OneDrive\Desktop\python fall detect\trained.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Webcam setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (96, 96))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)

    label_text = "Fall" if predicted_label == 0 else "Non-Fall"

    # Display result on frame
    cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255) if label_text == "Fall" else (0, 255, 0), 3)
    cv2.imshow("Real-Time Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
