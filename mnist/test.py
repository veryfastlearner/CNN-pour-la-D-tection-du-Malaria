import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# 1. LOAD your saved model
model = tf.keras.models.load_model('mnist_model.h5')  # or full path

# 2. LOAD your PNG (adjust path for your Downloads folder)
# Windows:
# img_path = r'C:\Users\YourName\Downloads\your_digit.png'
# Mac/Linux:
img_path = '/Users/Administrateur/Downloads/three.png'

img = Image.open(img_path).convert('L')  # 'L' = grayscale

# 3. RESIZE to 28x28 (MNIST size)
img = img.resize((28, 28))
print(img)
# 4. INVERT if needed (MNIST has white digits on black background)
# If your image has black digit on white background, uncomment:
# img = ImageOps.invert(img)

# 5. CONVERT to numpy array & normalize (0-255 → 0-1)
img_array = np.array(img) / 255.0
print(img_array)

# 6. RESHAPE to match model input: (batch=1, height=28, width=28, channels=1)
img_array = img_array.reshape(1, 28, 28, 1)
print(img_array)

# 7. PREDICT
prediction = model.predict(img_array)
digit = np.argmax(prediction)
confidence = np.max(prediction)

print(f"Predicted digit: {digit}")
print(f"Confidence: {confidence:.1%}")

# Optional: Show all probabilities
print("\nAll probabilities:")
for i, prob in enumerate(prediction[0]):
    print(f"  {i}: {prob:.2%}")