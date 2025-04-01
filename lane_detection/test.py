import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class LuaTorchModelConverter:
    @staticmethod
    def convert_t7_to_pytorch(t7_path):
        """
        Attempt to convert Lua Torch .t7 model to PyTorch
        
        :param t7_path: Path to .t7 model file
        :return: Converted PyTorch model
        """
        try:
            # Use torch.load with specific encoding
            lua_model = torch.load(t7_path, encoding='latin1')
            
            # Create a generic neural network structure
            class GenericLaneNet(nn.Module):
                def __init__(self):
                    super(GenericLaneNet, self).__init__()
                    # Basic architecture - modify as needed based on your specific model
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(64 * 112 * 112, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 2)  # Assuming binary lane detection
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            model = GenericLaneNet()
            
            # Copy weights if possible
            if hasattr(lua_model, 'state_dict'):
                model.load_state_dict(lua_model.state_dict())
            
            return model
        
        except Exception as e:
            print(f"Error converting .t7 model: {e}")
            return None

class LaneDetector:
    def __init__(self, model_path):
        """
        Initialize lane detection with model
        
        :param model_path: Path to lane detection model
        """
        # Attempt to load and convert model
        self.model = LuaTorchModelConverter.convert_t7_to_pytorch(model_path)
        
        if self.model is None:
            print("Falling back to traditional lane detection methods")
            self.fallback_detector = self._create_fallback_detector()
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def _create_fallback_detector(self):
        """
        Create a fallback lane detection method using traditional CV techniques
        """
        def detect_lanes_hough(frame):
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Region of interest
            height, width = frame.shape[:2]
            mask = np.zeros_like(edges)
            
            # Define region of interest - bottom 60% of the image
            polygon = np.array([
                [(0, height),
                 (width * 0.2, height * 0.4),
                 (width * 0.8, height * 0.4),
                 (width, height)]
            ], np.int32)
            
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Hough Transform
            lines = cv2.HoughLinesP(
                masked_edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=50, 
                minLineLength=100, 
                maxLineGap=50
            )
            
            # Draw lines
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            return frame
        
        return detect_lanes_hough

    def detect_lanes(self, frame):
        """
        Detect lanes using deep learning or fallback method
        
        :param frame: Input video frame
        :return: Frame with lane annotations
        """
        # If model conversion failed, use fallback
        if self.model is None:
            return self.fallback_detector(frame)
        
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Transform the image
        input_tensor = self.transform(pil_image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Process output (simplistic approach)
        lane_mask = torch.sigmoid(output).numpy()
        lane_mask = (lane_mask > 0.5).astype(np.uint8)
        
        # Overlay lanes on original frame
        lane_overlay = np.zeros_like(frame)
        lane_overlay[lane_mask[0, 0] > 0] = [0, 0, 255]  # Red lanes
        
        # Blend original frame with lane overlay
        annotated_frame = cv2.addWeighted(frame, 0.7, lane_overlay, 0.3, 0)
        
        return annotated_frame

def process_video(input_video_path, output_video_path, model_path):
    """
    Process entire video for lane detection
    
    :param input_video_path: Path to input video
    :param output_video_path: Path to save annotated video
    :param model_path: Path to lane detection model
    """
    # Validate input video and model
    if not os.path.exists(input_video_path):
        print(f"Error: Input video {input_video_path} not found!")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    # Initialize lane detector
    lane_detector = LaneDetector(model_path)
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video!")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and annotate lanes
        annotated_frame = lane_detector.detect_lanes(frame)
        
        # Write annotated frame
        out.write(annotated_frame)
        
        # Optional: Display progress
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
        
        # Optional: Display frame
        cv2.imshow('Lane Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to {output_video_path}")

def main():
    # Validate input arguments
    if len(sys.argv) < 4:
        print("Usage: python lane_detection.py <input_video> <output_video> <model_file>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    model_path = sys.argv[3]
    
    process_video(input_video, output_video, model_path)

if __name__ == '__main__':
    main()