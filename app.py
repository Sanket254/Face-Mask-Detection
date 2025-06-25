import streamlit as st
import cv2
import torch
import time
import os
import numpy as np
from PIL import Image

# Set paths
MODEL_PATH = 'C:/Users/sanke/Desktop/YOLO_FaceMask/yolov5/runs/train/exp4/weights/best.pt'
YOLO_PATH = 'C:/Users/sanke/Desktop/YOLO_FaceMask/yolov5'
SAVE_DIR = 'C:/Users/sanke/Desktop/NoMask_Captures'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model once
@st.cache_resource
def load_model():
    return torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local')

model = load_model()
class_names = ['NoMask', 'Mask']
cooldown_time = 10  # seconds
last_saved_times = []

def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

# UI
st.title("🛡️ Face Mask Detection Dashboard (YOLOv5)")
start = st.button("▶️ Start Detection")
stop = st.button("⏹️ Stop Detection")
frame_placeholder = st.empty()

if start and not stop:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.xyxy[0]
        current_time = time.time()
        updated_faces = []

        for *xyxy, conf, cls in detections:
            label_idx = int(cls.item())
            label = class_names[label_idx]
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf.item()*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if label == "NoMask":
                matched = False
                for i, (prev_box, last_time) in enumerate(last_saved_times):
                    if iou((x1, y1, x2, y2), prev_box) > 0.5:
                        matched = True
                        if current_time - last_time >= cooldown_time:
                            filename = f"no_mask_{int(current_time)}.jpg"
                            save_path = os.path.join(SAVE_DIR, filename)
                            cv2.imwrite(save_path, frame)
                            last_saved_times[i] = ((x1, y1, x2, y2), current_time)
                        else:
                            updated_faces.append((prev_box, last_time))
                        break

                if not matched:
                    filename = f"no_mask_{int(current_time)}.jpg"
                    save_path = os.path.join(SAVE_DIR, filename)
                    cv2.imwrite(save_path, frame)
                    updated_faces.append(((x1, y1, x2, y2), current_time))

        last_saved_times = updated_faces

        # Stream image in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    cap.release()
cv2.destroyAllWindows()

# Show saved NoMask images
st.subheader("📸 Captured 'NoMask' Frames")
image_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')], reverse=True)

if image_files:
    cols = st.columns(3)  # 3 images per row
    for i, file in enumerate(image_files):
        img_path = os.path.join(SAVE_DIR, file)
        img = Image.open(img_path)
        cols[i % 3].image(img, caption=file, use_column_width=True)
else:
    st.info("No images captured yet.")

