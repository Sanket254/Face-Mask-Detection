import cv2
import torch
import time
import os
import numpy as np

# Load YOLOv5 model
model = torch.hub.load(
    'C:/Users/sanke/Desktop/YOLO_FaceMask/yolov5',
    'custom',
    path=r'C:\Users\sanke\Desktop\YOLO_FaceMask\yolov5\runs\train\retrain_mask_model5\weights\best.pt',
    source='local'
)

class_names = ['NoMask', 'Mask']  # 0: NoMask, 1: Mask

save_dir = 'C:/Users/sanke/Desktop/NoMask_Captures'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

cooldown_time = 10  # seconds
last_saved_times = []  # to store last save time for each face

def iou(box1, box2):
    """Calculate IoU between two boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
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
                        save_path = os.path.join(save_dir, filename)
                        cv2.imwrite(save_path, frame)
                        last_saved_times[i] = ((x1, y1, x2, y2), current_time)
                    else:
                        updated_faces.append((prev_box, last_time))  # still cooling down
                    break

            if not matched:
                # New face without mask
                filename = f"no_mask_{int(current_time)}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, frame)
                updated_faces.append(((x1, y1, x2, y2), current_time))
        else:
            # Person now has mask → remove any cooldown matching this face
            for i, (prev_box, last_time) in enumerate(last_saved_times):
                if iou((x1, y1, x2, y2), prev_box) > 0.5:
                    # Cooldown reset
                    continue

    # Update face cooldowns
    last_saved_times = updated_faces

    cv2.imshow("YOLOv5 Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()