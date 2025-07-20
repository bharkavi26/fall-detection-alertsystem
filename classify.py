import tensorflow as tf
import numpy as np
import cv2

# ----------- Constants -----------
IMAGE_PATH = r'C:\Users\malai\OneDrive\Desktop\python fall detect\fall_0020.jpg'
           # ✅ Your test image file
MODEL_PATH = r'C:\Users\malai\OneDrive\Desktop\python fall detect\trained.tflite'     # ✅ TFLite model file
INPUT_SIZE = 96                   # ✅ Change if your model uses another size (check Edge Impulse)

# ----------- Load Image -----------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# ----------- Load TFLite Model -----------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (96, 96))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)


interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# ----------- Get Prediction -----------
output_data = interpreter.get_tensor(output_details[0]['index'])[0]
predicted_class = np.argmax(output_data)
print(predicted_class)
# ----------- Labels (Optional: add your labels if known) -----------
labels = ['fall', 'non-fall']  # Change if your model has different classes
print(f"Predicted class: {labels[predicted_class]} ({predicted_class})")
print(input_details[0]['shape'])  # Should give something like (1, 96, 96, 3)



