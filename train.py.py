import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Parameters
img_size = 64

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
    return images

def load_dataset(real_path, fake_path):
    real_imgs = load_images_from_folder(real_path)
    fake_imgs = load_images_from_folder(fake_path)
    X = np.array(real_imgs + fake_imgs)
    y = np.array([0]*len(real_imgs) + [1]*len(fake_imgs))
    return X, y

# Replace with your actual paths
real_path = 'D:/Deep fake project/your_dataset/Real'
fake_path = 'D:/Deep fake project/your_dataset/Fake'

X, y = load_dataset(real_path, fake_path)
X = X/255.0  # normalize pixel values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model Definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Dropout(0.2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
model.save('deepfake_model.keras')