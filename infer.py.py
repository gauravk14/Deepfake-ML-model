import cv2
import numpy as np
from tensorflow.keras.models import load_model

img_size = 64  # must match your model!
img_path = r"C:\Users\Asus\Desktop\fake_916.jpg"


model = load_model('deepfake_model.keras')

img = cv2.imread(img_path)
if img is None:
    print(f"Could not read image: {img_path}")
else:
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # shape (1, 64, 64, 3)
    print("Image shape for prediction:", img.shape)
    pred = model.predict(img)
    label = 'fake' if pred[0][0] > 0.5 else 'real'
    print(f'This image is predicted as: {label} (score={pred[0][0]:.2f})')
