import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from picamera import PiCamera
from picarx import Picarx
from time import sleep
import readchar
import threading

class LaneDetector:
    def __init__(self, model_path=None):
        """Initialize lane detection with a model or fallback method"""
        self.model = self.load_model(model_path) if model_path else None

        # Image transformation for model input
        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Load a pre-trained PyTorch model"""
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def detect_lanes(self, frame):
        """Detect lanes using AI model or OpenCV fallback"""
        if self.model:
            return self.detect_with_model(frame)
        else:
            return self.detect_with_opencv(frame)

    def detect_with_model(self, frame):
        """Lane detection using PyTorch model"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = self.model(img)  # Model inference
        lanes = output.squeeze().numpy()
        
        # Draw lanes (dummy example)
        cv2.line(frame, (100, 300), (500, 300), (0, 255, 0), 3)
        return frame

    def detect_with_opencv(self, frame):
        """Lane detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        mask = np.zeros_like(edges)
        height, width = frame.shape[:2]
        region = np.array([[(0, height), (width * 0.4, height * 0.5), (width * 0.6, height * 0.5), (width, height)]], np.int32)
        cv2.fillPoly(mask, region, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return frame

class PiCarController:
    def __init__(self, lane_detector):
        """Initialize PiCar-X with manual & autonomous driving"""
        pan_angle = 0
        tilt_angle = -15  # Set initial tilt angle to -15
        speed = 10
        px = Picarx()
        
        px.set_cam_tilt_angle(tilt_angle)  # Apply tilt angle at startup
        px.set_cam_pan_angle(pan_angle)
        self.lane_detector = lane_detector
        self.manual_mode = True  # Default mode: manual
        self.running = True

    def control_loop(self):
        """Main control loop for manual & autonomous driving"""
        while self.running:
            key = readchar.readkey().lower()
            if key in 'wsadikjl1m':
                self.handle_keypress(key)
            elif key == 'p':
                self.running = False  # Stop everything
            elif key == readchar.key.CTRL_C:
                self.running = False  # Force exit

    def handle_keypress(self, key):
        """Handle keyboard inputs"""
        speed = 10 if self.manual_mode else 30  # Autonomous mode is faster

        if key == 'm':  # Toggle manual/auto mode
            self.manual_mode = not self.manual_mode
            print("🚗 Mode:", "Manual" if self.manual_mode else "Autonomous")

        elif self.manual_mode:  # Manual control
            if key == 'w':
                self.px.forward(speed)
            elif key == 's':
                self.px.backward(speed)
            elif key == 'a':
                self.px.set_dir_servo_angle(-30)
                self.px.forward(speed)
            elif key == 'd':
                self.px.set_dir_servo_angle(30)
                self.px.forward(speed)

        elif not self.manual_mode:  # Autonomous mode
            self.autonomous_drive()

    def autonomous_drive(self):
        """Autonomous driving based on lane detection"""
        cap = cv2.VideoCapture(0)  # Open camera
        while not self.manual_mode and self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Lane detection
            annotated_frame = self.lane_detector.detect_lanes(frame)
            cv2.imshow("Lane Detection", annotated_frame)

            # Simple lane-following logic
            height, width, _ = frame.shape
            center_x = width // 2
            if center_x < width * 0.4:
                self.px.set_dir_servo_angle(-20)
            elif center_x > width * 0.6:
                self.px.set_dir_servo_angle(20)
            else:
                self.px.set_dir_servo_angle(0)

            self.px.forward(30)  # Move forward

            if cv2.waitKey(1) & 0xFF == ord('m'):  # Switch back to manual
                self.manual_mode = True

        cap.release()
        cv2.destroyAllWindows()

class VideoRecorder:
    def __init__(self):
        """Initialize PiCamera for recording"""
        self.camera = PiCamera()
        self.camera.resolution = (1280, 720)
        self.recording = False

    def start_recording(self):
        """Start recording in a new thread"""
        if not self.recording:
            self.recording = True
            self.camera.start_recording('video.h264')
            print("📹 Recording started...")

    def stop_recording(self):
        """Stop recording safely"""
        if self.recording:
            self.camera.stop_recording()
            print("✅ Video saved as 'video.h264'")
            self.recording = False

if __name__ == "__main__":
    lane_detector = LaneDetector()  # Load lane detector
    car = PiCarController(lane_detector)
    recorder = VideoRecorder()

    # Start Recording
    threading.Thread(target=recorder.start_recording).start()

    # Start Driving
    try:
        car.control_loop()
    finally:
        car.px.stop()
        recorder.stop_recording()
