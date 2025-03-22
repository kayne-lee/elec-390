import cv2
import torch
from ultralytics import YOLO
import argparse
import os

def detect_objects(input_image_path, output_image_path=None, confidence_threshold=0.25):
    """
    Detect objects in an image using YOLOv8 model and save the annotated image.
    
    Args:
        input_image_path (str): Path to the input image
        output_image_path (str, optional): Path to save the output image. If None, will use input_name_detected.jpg
        confidence_threshold (float, optional): Confidence threshold for detections (0-1)
    
    Returns:
        list: List of detection results
        str: Path to the saved annotated image
    """
    # Check if input image exists
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    # Create output path if not provided
    if output_image_path is None:
        base_name = os.path.basename(input_image_path)
        name, ext = os.path.splitext(base_name)
        output_image_path = os.path.join(os.path.dirname(input_image_path), f"{name}_detected{ext}")
    
    # Load the YOLOv8 model
    model = YOLO("best (4).pt")
    
    # Read the image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_image_path}")
    
    # Run YOLO model on the image
    results = model(image, conf=confidence_threshold)
    
    # Create a copy of the image for drawing
    annotated_image = image.copy()
    
    # Store detection information
    detections = []
    
    # Draw detections on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            class_name = model.names[cls]
            
            # Draw rectangle & label
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store detection info
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
    
    # Save the annotated image
    cv2.imwrite(output_image_path, annotated_image)
    
    return detections, output_image_path

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Detect objects in an image using YOLOv8")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", help="Path to output image (optional)")
    parser.add_argument("--conf", "-c", type=float, default=0.25, 
                        help="Confidence threshold (0-1)")
    
    args = parser.parse_args()
    
    # Run detection
    try:
        detections, output_path = detect_objects(
            args.input, 
            args.output, 
            args.conf
        )
        
        # Print results
        print(f"Detection complete! Found {len(detections)} objects.")
        print(f"Annotated image saved to: {output_path}")
        
        # Display the image (optional)
        image = cv2.imread(output_path)
        cv2.imshow("YOLOv8 Detection Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")