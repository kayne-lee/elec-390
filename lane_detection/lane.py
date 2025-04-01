import cv2
import numpy as np
import time

# Function to detect right lane
def detect_right_lane(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define the region of interest (ROI) to focus on the right lane
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Adjusted ROI to avoid right-side false detection
    polygon = np.array([[
        (int(width * 0.55), height),
        (int(width * 0.75), int(height * 0.6)),
        (int(width * 0.9), int(height * 0.6)),
        (int(width * 0.95), height)
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
    cap = cv2.VideoCapture(input_video)

    # Get FPS of video to time coordinate printing
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.25)  # Every 0.25 seconds
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect the right lane in the frame
        lines = detect_right_lane(frame)

        # Print lane coordinates every 0.25 seconds
        if frame_count % frame_interval == 0 and lines is not None:
            print(f"Lane Coordinates at {round(frame_count / fps, 2)} sec:")
            for line in lines:
                for x1, y1, x2, y2 in line:
                    print(f"  ({x1}, {y1}) -> ({x2}, {y2})")

        # Draw the detected lanes on the frame
        draw_lanes(frame, lines)

        # Display the frame
        cv2.imshow("Lane Detection", frame)

        frame_count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
input_video = "lane_detection/output.mp4"
process_video(input_video)

