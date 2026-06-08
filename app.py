import asyncio
import time
import json
import cv2
import numpy as np
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import os

# ---------------- CONFIGURATION ----------------
PORT = 8000
HOST = "0.0.0.0"
MODEL_SLEEP_PATH = os.path.join("models", "infant_sleep_position.pt")

# Motion tracking thresholds
INACTIVE_LIMIT = 5.0   # seconds for INACTIVE alert
LOW_LIMIT = 2.0        # seconds for LOW MOVEMENT
MOTION_THRESHOLD = 4.5 # mean pixel diff threshold

app = FastAPI(title="MatriSathi Backend - Strict Real-time Feed", version="4.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
connected_websockets = set()

class CameraWorker:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.camera_available = False
        
        # Load YOLO model 1: Sleep Position Detection
        print(f"Loading sleep position model from {MODEL_SLEEP_PATH}...")
        try:
            self.model_sleep = YOLO(MODEL_SLEEP_PATH)
            print("Sleep model loaded successfully!")
        except Exception as e:
            print(f"WARNING: Failed to load sleep model: {e}")
            print("  -> Sleep position detection will be unavailable.")
            self.model_sleep = None

        # Load YOLO model 2: Person Tracking / Motion Detection (yolov8s.pt as in motion.py)
        print("Loading motion/person tracking model yolov8s.pt...")
        try:
            self.model_motion = YOLO("yolov8s.pt")
            print("Motion model loaded successfully!")
        except Exception as e:
            print(f"WARNING: Failed to load motion model: {e}")
            print("  -> Motion/person tracking will be unavailable.")
            self.model_motion = None

        # Initialize Video Capture (graceful fallback if no camera)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("WARNING: Camera could not be opened. Live video feed will be unavailable.")
            print("  -> The web interface will still work but without live camera stream.")
            self.cap = None
            self.camera_available = False
        else:
            self.camera_available = True
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 10-second sleep model throttling variables
        self.last_sleep_inference_time = 0.0
        self.last_sleep_results = None
        self.sleep_detected = False
        self.sleep_position = "Unknown"
        self.sleep_confidence = 0.0
        self.sleep_box = None

        # Combined Monitoring state
        init_msg = "Matri Sathi Monitoring System Initialized"
        if not self.camera_available:
            init_msg += " (No camera detected — live feed unavailable)"
        self.state = {
            "infant_detected": False,
            "motion": {
                "status": "ACTIVE",
                "score": 0.0,
                "inactive_seconds": 0.0,
                "box": None
            },
            "sleep": {
                "position": "Unknown",
                "confidence": 0.0,
                "box": None
            },
            "is_simulation": False,
            "camera_available": self.camera_available,
            "logs": [
                {"time": self.get_time_str(), "type": "info", "msg": init_msg}
            ]
        }

        # Motion detection structures
        self.track_data = {}
        self.last_log_state = {
            "position": None,
            "motion_status": None,
            "detected": False
        }

        # Start background loop only if camera is available
        if self.camera_available and (self.model_sleep or self.model_motion):
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        else:
            self.thread = None
            print("Camera loop not started (no camera or no models available).")

    def get_time_str(self):
        return time.strftime("%H:%M:%S")

    def add_log(self, type_str, msg):
        log_entry = {"time": self.get_time_str(), "type": type_str, "msg": msg}
        self.state["logs"].insert(0, log_entry)
        if len(self.state["logs"]) > 20:
            self.state["logs"].pop()
        
        # Broadcast immediately via WebSockets if any are active
        asyncio.run_coroutine_threadsafe(
            self.broadcast_state(),
            loop=main_loop
        )

    async def broadcast_state(self):
        if not connected_websockets:
            return
        payload = json.dumps(self.state)
        for ws in list(connected_websockets):
            try:
                await ws.send_text(payload)
            except Exception:
                connected_websockets.remove(ws)

    def run(self):
        print("CameraWorker loop started.")

        while self.running:
            loop_start = time.time()

            if not self.camera_available or self.cap is None:
                time.sleep(1)
                continue

            # Read from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("WARNING: Failed to grab webcam frame. Retrying...")
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---------------- MODEL 1: PERSON TRACKING & MOTION Surveillance ----------------
            if self.model_motion is None:
                results_motion = []
            else:
                results_motion = self.model_motion.track(frame, persist=True, verbose=False)
            
            person_detected = False
            motion_box = None
            motion_score = 0.0
            inactive_time = 0.0
            motion_status = "UNKNOWN"
            
            active_count = 0
            low_count = 0
            inactive_count = 0

            if len(results_motion) > 0 and results_motion[0].boxes is not None:
                for box in results_motion[0].boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    
                    # COCO 'person' (class 0)
                    if cls == 0 and conf >= 0.5:
                        person_detected = True
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        motion_box = [x1, y1, x2, y2]

                        roi = gray[y1:y2, x1:x2]
                        if roi.size > 0:
                            roi = cv2.GaussianBlur(roi, (5, 5), 0)
                            person_id = int(box.id.item()) if box.id is not None else 0

                            if person_id not in self.track_data:
                                self.track_data[person_id] = {
                                    "prev_roi": roi,
                                    "last_move": time.time(),
                                    "history": []
                                }

                            prev_roi = self.track_data[person_id]["prev_roi"]
                            roi_resized = cv2.resize(roi, (prev_roi.shape[1], prev_roi.shape[0]))
                            
                            diff = cv2.absdiff(prev_roi, roi_resized)
                            motion_score = float(np.mean(diff))

                            self.track_data[person_id]["history"].append(motion_score)
                            if len(self.track_data[person_id]["history"]) > 10:
                                self.track_data[person_id]["history"].pop(0)
                            avg_motion = float(np.mean(self.track_data[person_id]["history"]))

                            if avg_motion > MOTION_THRESHOLD:
                                self.track_data[person_id]["last_move"] = time.time()

                            inactive_time = time.time() - self.track_data[person_id]["last_move"]
                            self.track_data[person_id]["prev_roi"] = roi
                            motion_score = avg_motion
                        else:
                            inactive_time = 0.0

                        if inactive_time >= INACTIVE_LIMIT:
                            motion_status = "INACTIVE"
                            color_m = (0, 0, 255) # Red
                            inactive_count += 1
                        elif inactive_time >= LOW_LIMIT:
                            motion_status = "LOW MOVEMENT"
                            color_m = (0, 255, 255) # Yellow
                            low_count += 1
                        else:
                            motion_status = "ACTIVE"
                            color_m = (0, 255, 0) # Green
                            active_count += 1

                        # Draw solid bounding box for motion status
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_m, 2)
                        cv2.putText(
                            frame,
                            f"BABY ID {person_id}: {motion_status}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color_m,
                            2
                        )

            # ---------------- MODEL 2: SLEEP POSITION CLASSIFIER (Throttled to every 10 seconds) ----------------
            current_time = time.time()
            if current_time - self.last_sleep_inference_time >= 10.0:
                if self.model_sleep is not None:
                    print(f"[{self.get_time_str()}] Running sleep position YOLO model on live feed (10s interval)...")
                    self.last_sleep_results = self.model_sleep(frame, verbose=False)
                else:
                    self.last_sleep_results = None
                self.last_sleep_inference_time = current_time

                # Parse the inference outputs inside the 10-second block
                sleep_detected = False
                pos_label = "Unknown"
                conf_val = 0.0
                sleep_box = None
                
                if self.last_sleep_results is not None and len(self.last_sleep_results) > 0 and self.last_sleep_results[0].boxes is not None:
                    max_conf = -1.0
                    for box in self.last_sleep_results[0].boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        
                        # 0: supine (Back), 1: prone (Prone)
                        if cls in [0, 1] and conf > 0.4:
                            if conf > max_conf:
                                max_conf = conf
                                sleep_detected = True
                                pos_label = "Back" if cls == 0 else "Prone"
                                conf_val = conf
                                
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(frame.shape[1], x2)
                                y2 = min(frame.shape[0], y2)
                                sleep_box = [x1, y1, x2, y2]
                                
                # Cache results to instance variables
                self.sleep_detected = sleep_detected
                self.sleep_position = pos_label
                self.sleep_confidence = conf_val
                self.sleep_box = sleep_box

                # Print outputs directly to the server console every 10 seconds
                print(f"[{self.get_time_str()}] Sleep Classifier Output: position='{self.sleep_position}', confidence={self.sleep_confidence*100:.2f}%, box={self.sleep_box}")

            # Draw top-left dashboard overlays (motion.py console printouts)
            cv2.putText(frame, f"ACTIVE: {active_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"LOW: {low_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, f"INACTIVE: {inactive_count}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
            if inactive_count > 0:
                cv2.putText(frame, "ALERT: INACTIVE DETECTED", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ---------------- STATE SYNCHRONIZATION ----------------
            detected = person_detected or self.sleep_detected
            
            with self.lock:
                self.state["infant_detected"] = detected
                self.state["motion"] = {
                    "status": motion_status if person_detected else "UNKNOWN",
                    "score": float(motion_score),
                    "inactive_seconds": float(inactive_time),
                    "box": motion_box
                }
                self.state["sleep"] = {
                    "position": self.sleep_position,
                    "confidence": float(self.sleep_confidence),
                    "box": self.sleep_box
                }
                self.latest_frame = frame

            self.check_and_log_state_changes(detected, self.sleep_position, motion_status, inactive_time)

            # Broadcast combined state to WebSocket
            asyncio.run_coroutine_threadsafe(
                self.broadcast_state(),
                loop=main_loop
            )

            # Throttle loop (~20 FPS)
            elapsed = time.time() - loop_start
            sleep_time = max(0.01, 0.05 - elapsed)
            time.sleep(sleep_time)

    def check_and_log_state_changes(self, detected, position, status, inactive_sec):
        was_detected = self.last_log_state.get("detected", False)
        if detected and not was_detected:
            self.add_log("info", "Infant detected in monitoring field.")
        elif not detected and was_detected:
            self.add_log("info", "Infant left the monitoring field.")
        
        if detected:
            if self.last_log_state["position"] != position and position != "Unknown":
                if position == "Back":
                    self.add_log("ok", "Sleep position: Back (Safe)")
                elif position == "Prone":
                    self.add_log("warn", "ALERT: Infant rolled onto stomach (Prone)!")
            
            if self.last_log_state["motion_status"] != status and status != "UNKNOWN":
                if status == "INACTIVE":
                    self.add_log("warn", f"Inactivity alert: No movement for {int(inactive_sec)}s.")
                elif status == "LOW MOVEMENT":
                    self.add_log("info", "Low infant movement detected.")
                elif status == "ACTIVE":
                    self.add_log("ok", "Infant is active.")

        self.last_log_state["position"] = position
        self.last_log_state["motion_status"] = status
        self.last_log_state["detected"] = detected

    def generate_mjpeg(self):
        while self.running:
            with self.lock:
                if self.latest_frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    if not self.camera_available:
                        cv2.putText(frame, "NO CAMERA DETECTED", (155, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                        cv2.putText(frame, "Connect a camera and restart", (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                    else:
                        cv2.putText(frame, "LOADING STREAM...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', frame)
                else:
                    _, jpeg = cv2.imencode('.jpg', self.latest_frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.05)  # 20 FPS

    def shutdown(self):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        print("CameraWorker shut down.")

camera_worker = None
main_loop = None

@app.on_event("startup")
def startup_event():
    global camera_worker, main_loop
    main_loop = asyncio.get_running_loop()
    try:
        camera_worker = CameraWorker()
    except Exception as e:
        print(f"WARNING: CameraWorker failed to initialize: {e}")
        print("  -> Server will run without camera features.")
        camera_worker = None

@app.on_event("shutdown")
def shutdown_event():
    if camera_worker:
        camera_worker.shutdown()

@app.get("/")
def get_index():
    return FileResponse("Index.html")

@app.get("/Index.html")
def get_index_alias():
    return FileResponse("Index.html")

@app.get("/sleep_position")
@app.get("/sleep_position.html")
def get_sleep_position():
    return FileResponse("sleep_position.html")

@app.get("/about")
@app.get("/about.html")
def get_about():
    return FileResponse("about.html")

@app.post("/api/predict_sleep")
async def predict_sleep(file: UploadFile = File(...)):
    if camera_worker is None or camera_worker.model_sleep is None:
        return {"error": "Sleep position model is not available. Make sure 'models/infant_sleep_position.pt' exists."}
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid image format"}
        
        # Run inference using the loaded sleep model from camera_worker
        results = camera_worker.model_sleep(img, verbose=False)
        
        predictions = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                
                # Check for class 0 (Back/supine) and 1 (Prone)
                if cls in [0, 1]:
                    pos_label = "Back" if cls == 0 else "Prone"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    predictions.append({
                        "position": pos_label,
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2]
                    })
        
        return {"predictions": predictions}
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

@app.get("/video_feed")
def video_feed():
    if camera_worker is None:
        # Return a single placeholder frame if no camera worker
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "BACKEND RUNNING - NO CAMERA", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        _, jpeg = cv2.imencode('.jpg', placeholder)
        def single_frame():
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                time.sleep(1)
        return StreamingResponse(single_frame(), media_type="multipart/x-mixed-replace; boundary=frame")
    return StreamingResponse(
        camera_worker.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# WebSocket Handler
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    print(f"WebSocket client connected. Total clients: {len(connected_websockets)}")
    
    if camera_worker:
        try:
            await websocket.send_text(json.dumps(camera_worker.state))
        except Exception as e:
            print(f"Error sending initial state: {e}")
            
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
        print(f"WebSocket client disconnected. Total clients: {len(connected_websockets)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

# Serve static folders
if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Run app
if __name__ == "__main__":
    import uvicorn
    import socket

    def get_local_ip():
        """Get the LAN IP address of this machine."""
        try:
            # Connect to an external address to determine which interface is used
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()
    print("\n" + "=" * 60)
    print("  MatriSathi Backend Starting...")
    print("=" * 60)
    print(f"  Local (this device):  http://localhost:{PORT}")
    print(f"  Network (other devices): http://{local_ip}:{PORT}")
    print("-" * 60)
    print("  NOTE: If other devices cannot connect, you may need to")
    print(f"  allow port {PORT} through Windows Firewall:")
    print(f"    netsh advfirewall firewall add rule name=\"MatriSathi\"")
    print(f"    dir=in action=allow protocol=TCP localport={PORT}")
    print("=" * 60 + "\n")

    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
