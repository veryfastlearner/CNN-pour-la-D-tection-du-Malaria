import tensorflow as tf
import numpy as np
from PIL import Image
import sys

# 1. LOAD trained model
model = tf.keras.models.load_model('malaria_mini.h5')

# 2. LOAD test image (provide path as argument or use default)
image_path = '/Users/Administrateur/Downloads/cell.jpeg'

# 3. PREPROCESS like training
img = Image.open(image_path).convert('RGB').resize((64, 64))
img_array = np.array(img).reshape(1, 64, 64, 3) / 255.0

# 4. PREDICT
prediction = model.predict(img_array, verbose=0)
confidence = prediction[0][0]

# 5. OUTPUT
print(f"\n🔬 Malaria Detection Result:")
print(f"   Image: {image_path}")
print(f"   Infection probability: {confidence:.1%}")
if confidence > 0.5:
    print(f"   Diagnosis: 🔴 PARASITIZED (malaria positive)")
else:
    print(f"   Diagnosis: 🟢 UNINFECTED (malaria negative)")