import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO('yolov8n.pt')  # You can also use yolov8s.pt for better accuracy

# Open webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

# COCO class ID for person is 0
person_class_id = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run detection
    results = model(frame)[0]

    for result in results.boxes:
        cls = int(result.cls[0])
        conf = float(result.conf[0])

        if cls == person_class_id:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            label = model.names[cls]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Print detection message
            print(f"[INFO] Person detected with confidence {conf:.2f}")

    # Display the frame
    cv2.imshow("Person Detection", frame)

    # Press spacebar to exit
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

cap.release()
cv2.destroyAllWindows()
