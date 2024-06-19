import cv2
from ultralytics import YOLO

import sys

if len(sys.argv) == 2:
    video = sys.argv[1]
    brightness = 255
elif len(sys.argv) == 3:
    video = sys.argv[1]
    brightness = 255

    if sys.argv[2] == "bright":
        brightness = 400
    elif sys.argv[2] == "normal":
        brightness = 255
    elif sys.argv[2] == "dark":
        brightness = 150
else:
    video = "rowerzysci.mp4"
    brightness = 255

# Załaduj model YOLOv8
model = YOLO('yolov8n.pt')

# Funkcja do rysowania wykrytych obiektów na obrazie
def draw_detections(image, detections):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['box'])
        confidence = detection['confidence']
        class_id = detection['class_id']
        label = f"{model.names[class_id]} {confidence:.2f}"

        if class_id == 0: # persons
            colors = (255, 0, 0)
        elif class_id == 1: # bicycles
            colors = (0, 0, 255)
        else: #cars
            colors = (0, 255, 0)
        
        # Rysowanie prostokąta i etykiety
        cv2.rectangle(image, (x1, y1), (x2, y2), colors, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors, 1)

# Otwórz wideo
cap = cv2.VideoCapture(video)

classes = ['car', 'person', 'truck', 'bicycle', 'bus']

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (960, 540))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        break
      
    # Wykonaj detekcję na bieżącej klatce
    results = model.predict(frame, verbose=False)
    
    # Przetwarzanie wyników
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # współrzędne skrzynek
        confidences = result.boxes.conf.cpu().numpy()  # współczynniki ufności
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # id klasy

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if model.names[class_id] in classes:
                detections.append({
                    'box': box,
                    'confidence': conf,
                    'class_id': class_id
                })
    
    # Rysowanie wykryć na klatce
    cv2.normalize(frame, frame, 0, brightness, cv2.NORM_MINMAX)
    draw_detections(frame, detections)
    
    # Konwersja koloru z BGR (OpenCV) na RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Wyświetlanie obrazu
    cv2.imshow("Frame", frame_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()
