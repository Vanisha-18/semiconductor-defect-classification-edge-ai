import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random

print("Creating demo video...")

# Load model
model = tf.keras.models.load_model('iesa_model_BEST.h5')
class_names = ['Bridges', 'CMP_Scratches', 'Clean', 'Cracks', 'LER', 'Malformed_Vias', 'Opens', 'Others']

# Get sample test images
test_dir = 'Test'
sample_images = []
for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    if os.path.exists(class_path):
        imgs = [os.path.join(class_path, f) for f in os.listdir(class_path)[:3]]
        sample_images.extend(imgs)

random.shuffle(sample_images)
sample_images = sample_images[:20]

# Video settings
fps = 2
width, height = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('iesa_demo.mp4', fourcc, fps, (width, height))

print(f"Processing {len(sample_images)} images...")

for idx, img_path in enumerate(sample_images):
    # Load and predict
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    # Create frame
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Title (using basic font - cv2.FONT_HERSHEY_SIMPLEX)
    cv2.putText(frame, "IESA DeepTech Hackathon 2026", (50, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(frame, "Edge AI Defect Classification System", (50, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    # Original image
    img_display = cv2.cvtColor(np.array(img.resize((400, 400))), cv2.COLOR_RGB2BGR)
    frame[200:600, 100:500] = img_display
    cv2.putText(frame, "Input Image", (100, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Prediction box
    cv2.rectangle(frame, (600, 200), (1150, 600), (0, 0, 0), 3)
    cv2.putText(frame, "Prediction Results", (650, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Results
    color = (0, 200, 0) if confidence > 60 else (0, 165, 255)
    cv2.putText(frame, f"Class: {pred_class}", (650, 330), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (650, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Progress bar
    bar_width = int(400 * confidence / 100)
    cv2.rectangle(frame, (650, 430), (650 + bar_width, 460), color, -1)
    cv2.rectangle(frame, (650, 430), (1050, 460), (0, 0, 0), 2)
    
    # Model info
    cv2.putText(frame, "Model: MobileNetV2", (650, 520), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    cv2.putText(frame, "Size: 9.11 MB | Accuracy: 69.93%", (650, 555), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    
    # Frame counter
    cv2.putText(frame, f"Sample {idx+1}/{len(sample_images)}", (50, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Write frames
    for _ in range(fps):
        video.write(frame)
    
    print(f"  Frame {idx+1}/{len(sample_images)}: {pred_class} ({confidence:.1f}%)")

video.release()
print("\nVideo created: iesa_demo.mp4")

# Download
from google.colab import files
files.download('iesa_demo.mp4')

print("Video downloading!")
