from ultralytics import YOLO
import cv2
import math
import numpy as np
import pyttsx3, threading, time

def speak_async(text):
    def worker():
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 170)
        local_engine.say(text)
        local_engine.runAndWait()
    threading.Thread(target=worker, daemon=True).start()


# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Load your trained YOLO model
model = YOLO("F:/Diploma/runs/detect/train6/weights/best.pt")

classNames = model.names


last_time = 0
interval = 3

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence and class
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Find center of bounding box
            cx = int((x1 + x2) / 2)

            frame_center = img.shape[1] // 2
            if cx < frame_center - 200:
                position = "left"
            elif cx > frame_center + 200:
                position = "right"
            else:
                position = "center"


            #draw text on bounding box
            label = f"{class_name} ({position} {confidence})"
            org = (x1, y1 - 10)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            #tts phrase
            phrase = f"{class_name} {position}"
            current_time = time.time()
            if current_time - last_time > interval:
                speak_async(phrase)
                last_time = current_time


    # Show frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
