import cv2
from ultralytics import YOLO

# Load a YOLOv8 model (you can also use yolov8s.pt or yolov8n.pt for faster detection)
model = YOLO('yolov8n.pt')  # Tiny model for better speed; can also use 'yolov8m.pt' or 'yolov8l.pt'

# Open the webcam (change index to match your camera or use a video file)
cap = cv2.VideoCapture(0)

# Vehicle classes in COCO dataset: car=2, motorcycle=3, bus=5, truck=7
vehicle_classes = [2, 3, 5, 7]

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    results = model(frame)[0]

    for result in results.boxes:
        cls = int(result.cls[0])
        conf = float(result.conf[0])
        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Vehicle Detection", frame)

    # Press spacebar to exit
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

cap.release()
cv2.destroyAllWindows()
