import base64
import io
import json
from typing import List, Dict, Any
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names


def estimate_position(cx: int, width: int, margin: int = 100) -> str:
    center = width // 2
    if cx < center - margin:
        return "left"
    elif cx > center + margin:
        return "right"
    else:
        return "center"


def run_detection(frame_bgr: np.ndarray) -> Dict[str, Any]:
    results = model(frame_bgr, stream=True)
    h, w = frame_bgr.shape[:2]

    detections: List[Dict[str, Any]] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]

            cx = int((x1 + x2) / 2)
            position = estimate_position(cx, w)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": round(conf, 3),
                "position": position
            })
    return {"width": w, "height": h, "detections": detections}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)

            data_url = payload.get("image")
            if not data_url or "," not in data_url:
                await websocket.send_text(json.dumps({"error": "invalid_image"}))
                continue

            header, b64data = data_url.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_text(json.dumps({"error": "decode_failed"}))
                continue

            result = run_detection(frame_bgr)
            await websocket.send_text(json.dumps(result))
    except Exception as e:

        await websocket.close()
