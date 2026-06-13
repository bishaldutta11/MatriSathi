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
CRY_MODEL_PATH = os.path.join("models", "infant_cry_detect.keras")
cry_model = None
cry_model_loaded = False

def load_custom_cry_model(model_path):
    import zipfile
    import h5py
    import tempfile
    import keras
    from keras import layers
    import numpy as np

    # 1. Build the functional architecture matching the original Keras 2.15 structure in Keras 3
    inputs = keras.Input(shape=(310,), name="features")
    
    x = layers.BatchNormalization(axis=1, name="batch_normalization")(inputs)
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense")(x)
    x = layers.BatchNormalization(axis=1, name="batch_normalization_1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.BatchNormalization(axis=1, name="batch_normalization_2")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense_2")(x)
    x = layers.BatchNormalization(axis=1, name="batch_normalization_3")(x)
    x = layers.Dropout(0.4, name="dropout_2")(x)
    
    x = layers.Dense(64, activation="relu", name="dense_3")(x)
    outputs = layers.Dense(10, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="InfantCry_DNN")

    # 2. Extract model.weights.h5 and load into the model
    with zipfile.ZipFile(model_path, 'r') as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_tmp_path = z.extract("model.weights.h5", path=tmpdir)
            
            # Sequence of groups in HDF5 file containing weights
            h5_groups = [
                "layers\\batch_normalization/vars",
                "layers\\dense/vars",
                "layers\\batch_normalization_1/vars",
                "layers\\dense_1/vars",
                "layers\\batch_normalization_2/vars",
                "layers\\dense_2/vars",
                "layers\\batch_normalization_3/vars",
                "layers\\dense_3/vars",
                "layers\\dense_4/vars"
            ]
            
            # Layers in model that have weights
            model_layers = [
                model.get_layer("batch_normalization"),
                model.get_layer("dense"),
                model.get_layer("batch_normalization_1"),
                model.get_layer("dense_1"),
                model.get_layer("batch_normalization_2"),
                model.get_layer("dense_2"),
                model.get_layer("batch_normalization_3"),
                model.get_layer("dense_3"),
                model.get_layer("output")
            ]
            
            with h5py.File(weights_tmp_path, 'r') as f:
                for g_idx, layer in enumerate(model_layers):
                    # Check both backslash and forward slash styles in case of different OS saves
                    g_name_bs = h5_groups[g_idx]
                    g_name_fs = g_name_bs.replace("\\", "/")
                    
                    if g_name_bs in f:
                        group = f[g_name_bs]
                    elif g_name_fs in f:
                        group = f[g_name_fs]
                    else:
                        raise KeyError(f"Could not find weights group for layer {layer.name} in h5 file.")
                    
                    ds_keys = sorted(group.keys(), key=int)
                    weights = [np.array(group[k]) for k in ds_keys]
                    layer.set_weights(weights)
                    
    return model

@app.on_event("startup")
def startup_event():
    global camera_worker, main_loop, cry_model, cry_model_loaded
    main_loop = asyncio.get_running_loop()
    try:
        camera_worker = CameraWorker()
    except Exception as e:
        print(f"WARNING: CameraWorker failed to initialize: {e}")
        print("  -> Server will run without camera features.")
        camera_worker = None

    # Load Infant Cry Model on startup
    try:
        import keras
        import h5py
        keras_available = True
    except ImportError:
        keras_available = False

    if keras_available:
        if os.path.exists(CRY_MODEL_PATH):
            print(f"Loading infant cry detection model from {CRY_MODEL_PATH} on startup...")
            try:
                cry_model = load_custom_cry_model(CRY_MODEL_PATH)
                cry_model_loaded = True
                print("Infant cry detection model loaded successfully on startup!")
            except Exception as e:
                print(f"WARNING: Failed to load cry model on startup: {e}")
        else:
            print(f"WARNING: Cry model file not found at {CRY_MODEL_PATH}")
    else:
        print("INFO: Keras/TensorFlow/h5py not installed. Cry detection will run in simulation mode.")

    # Specify in terminal whether the cry model is loaded or demo model is used
    print("\n" + "=" * 60)
    if cry_model_loaded:
        print("  CRY DETECTION MODEL STATUS: [LOADED]")
        print(f"  Model File: {CRY_MODEL_PATH}")
        print("  Real AI inference will be used for infant cry diagnostics.")
    else:
        print("  CRY DETECTION MODEL STATUS: [DEMO / SIMULATION MODE]")
        print("  Using simulated prediction logic.")
    print("=" * 60 + "\n")


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

@app.get("/cry")
@app.get("/cry.html")
def get_cry():
    return FileResponse("cry.html")

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

# Infant Cry Model endpoint

@app.post("/api/predict_cry")
async def predict_cry(file: UploadFile = File(...)):
    global cry_model, cry_model_loaded
    
    # 10 infant cry categories
    classes = [
        "Hunger",
        "Pain / Colic",
        "Tiredness",
        "Discomfort / Wet Diaper",
        "Gas / Needs Burping",
        "Boredom / Seeking Attention",
        "Fear / Startled",
        "Temperature (Too Hot/Cold)",
        "Sickness / Fever",
        "Frustration / Overstimulation"
    ]
    
    # Check if Keras, Librosa, and h5py libraries are installed
    try:
        import librosa
        import keras
        import h5py
        libs_available = True
    except ImportError:
        libs_available = False

    contents = await file.read()
    
    # If Keras, Librosa, and h5py are available, and the model exists, perform real inference
    if libs_available and os.path.exists(CRY_MODEL_PATH):
        try:
            # Lazy load the Keras model once and cache it in memory
            if not cry_model_loaded:
                print(f"Loading infant cry detection model from {CRY_MODEL_PATH}...")
                cry_model = load_custom_cry_model(CRY_MODEL_PATH)
                cry_model_loaded = True
                print("Infant cry detection model loaded successfully!")
            
            # Load audio using librosa
            import io
            audio_file = io.BytesIO(contents)
            y, sr = librosa.load(audio_file, sr=None)
            
            # 1. Extract MFCC (20 coefficients): mean, std, min, max, median -> 100 features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_min = np.min(mfcc, axis=1)
            mfcc_max = np.max(mfcc, axis=1)
            mfcc_median = np.median(mfcc, axis=1)
            
            # 2. Extract Chroma STFT (12 coefficients): mean, std, min, max -> 48 features
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
                chroma_mean = np.mean(chroma, axis=1)
                chroma_std = np.std(chroma, axis=1)
                chroma_min = np.min(chroma, axis=1)
                chroma_max = np.max(chroma, axis=1)
            except Exception:
                chroma_mean = np.zeros(12)
                chroma_std = np.zeros(12)
                chroma_min = np.zeros(12)
                chroma_max = np.zeros(12)
                
            # 3. Extract Mel Spectrogram (128 coefficients): mean -> 128 features
            try:
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_mean = np.mean(mel, axis=1)
            except Exception:
                mel_mean = np.zeros(128)
                
            # 4. Extract Spectral Contrast (7 coefficients): mean, std, min, max -> 28 features
            try:
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
                contrast_mean = np.mean(contrast, axis=1)
                contrast_std = np.std(contrast, axis=1)
                contrast_min = np.min(contrast, axis=1)
                contrast_max = np.max(contrast, axis=1)
            except Exception:
                contrast_mean = np.zeros(7)
                contrast_std = np.zeros(7)
                contrast_min = np.zeros(7)
                contrast_max = np.zeros(7)
                
            # 5. Extract Tonnetz (6 coefficients): mean -> 6 features
            try:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                tonnetz_mean = np.mean(tonnetz, axis=1)
            except Exception:
                tonnetz_mean = np.zeros(6)
                
            # Concatenate all features to form the exact 310 features expected by the model
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_min, mfcc_max, mfcc_median,
                chroma_mean, chroma_std, chroma_min, chroma_max,
                mel_mean,
                contrast_mean, contrast_std, contrast_min, contrast_max,
                tonnetz_mean
            ])
            
            # Reshape features for model input (shape: [1, 310])
            features = np.expand_dims(features, axis=0)
            
            # Run prediction
            prediction = cry_model.predict(features, verbose=0)[0]
            
            # Format results
            predictions_list = []
            for idx, prob in enumerate(prediction):
                predictions_list.append({
                    "label": classes[idx],
                    "confidence": float(prob)
                })
            
            # Sort descending by confidence
            predictions_list.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "success": True,
                "is_simulated": False,
                "predictions": predictions_list,
                "primary": predictions_list[0]
            }
        except Exception as e:
            print(f"Error during real cry model prediction: {e}")
            # Fallthrough to simulation fallback on error
            pass

    # High-fidelity simulated prediction (deterministic based on file hash)
    import hashlib
    file_hash = int(hashlib.md5(contents).hexdigest(), 16)
    np_rand = np.random.RandomState(file_hash % (2**32 - 1))
    
    # Generate random probabilities summing to 1
    raw_scores = np_rand.rand(10)
    # Weight certain common cry causes slightly higher for realism
    raw_scores[0] *= 2.5  # Hunger
    raw_scores[1] *= 1.8  # Pain
    raw_scores[2] *= 1.5  # Tiredness
    raw_scores[3] *= 1.4  # Discomfort
    
    probabilities = raw_scores / np.sum(raw_scores)
    
    predictions_list = []
    for idx, prob in enumerate(probabilities):
        predictions_list.append({
            "label": classes[idx],
            "confidence": float(prob)
        })
    
    predictions_list.sort(key=lambda x: x["confidence"], reverse=True)
    
    warning_msg = (
        "Running in simulation mode. Please install 'tensorflow' and 'librosa' "
        "and place 'models/infant_cry_detect.keras' in the models folder to run local AI inference."
        if not libs_available else "Model inference failed, running in simulation mode."
    )
    
    return {
        "success": True,
        "is_simulated": True,
        "predictions": predictions_list,
        "primary": predictions_list[0],
        "warning": warning_msg
    }

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
