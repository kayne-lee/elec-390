import cv2
import numpy as np

# Function to detect right lane
def detect_right_lane(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define the region of interest (ROI) to focus on lower part of the frame
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.5), height),
        (int(width * 0.9), int(height * 0.6)),
        (width, int(height * 0.6)),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    
    # Mask the edges image to focus on the region of interest
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform to find lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    
    return lines

# Function to draw lanes on the frame
def draw_lanes(frame, lines):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

# Main function to process the video
def process_video(input_video):
    # Open the video
    cap = cv2.VideoCapture(input_video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect the right lane in the frame
        lines = detect_right_lane(frame)
        
        # Draw the detected lanes on the frame
        draw_lanes(frame, lines)
        
        # Display the frame
        cv2.imshow("Lane Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
input_video = "video/sr.mp4"  # Replace with the path to your video file
process_video(input_video)
