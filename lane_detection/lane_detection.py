import cv2
import numpy as np
import math
import time

# Global variables to track previous lane lines
prev_left_line = None
prev_right_line = None
smooth_factor = 0.8  # 80% previous, 20% current

# Function to mask out the region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function to draw the filled polygon between the lane lines
def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    
    # Fill the polygon between the lines
    cv2.fillPoly(line_img, poly_pts, color)
    
    # Overlay the polygon onto the original image
    img = cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)
    return img

# The lane detection pipeline
def pipeline(image):
    global prev_left_line, prev_right_line
    
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    # Convert to grayscale and apply Canny edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Use adaptive thresholding for better handling of lighting variations
    cannyed_image = cv2.Canny(blurred, 50, 150)

    # Mask out the region of interest
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    # Perform Hough Line Transformation to detect lines
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    # Separating left and right lines based on slope
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is None:
        # If no lines detected, use previous lines if available
        if prev_left_line is not None and prev_right_line is not None:
            return draw_lane_lines(image, prev_left_line, prev_right_line)
        return image

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if math.fabs(slope) < 0.5:  # Ignore nearly horizontal lines
                continue
            if slope <= 0:  # Left lane
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # Right lane
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Fit a quadratic polynomial for curved lanes
    min_y = int(image.shape[0] * (3 / 5))  # Slightly below the middle of the image
    max_y = image.shape[0]  # Bottom of the image

    # Try to detect and handle the left lane
    if len(left_line_x) > 1 and len(left_line_y) > 1:
        if len(left_line_x) >= 5:  # If enough points, fit quadratic curve
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=2))
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
        else:  # Otherwise, fit linear
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
        
        # Apply smoothing if we have previous lines
        if prev_left_line is not None:
            left_x_start = int(smooth_factor * prev_left_line[0] + (1-smooth_factor) * left_x_start)
            left_x_end = int(smooth_factor * prev_left_line[2] + (1-smooth_factor) * left_x_end)
    else:
        # Use previous line if available, otherwise default
        if prev_left_line is not None:
            left_x_start, _, left_x_end, _ = prev_left_line
        else:
            left_x_start, left_x_end = 0, 0

    # Try to detect and handle the right lane (similar logic)
    if len(right_line_x) > 1 and len(right_line_y) > 1:
        if len(right_line_x) >= 5:  # If enough points, fit quadratic curve
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=2))
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
        else:  # Otherwise, fit linear
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
        
        # Apply smoothing if we have previous lines
        if prev_right_line is not None:
            right_x_start = int(smooth_factor * prev_right_line[0] + (1-smooth_factor) * right_x_start)
            right_x_end = int(smooth_factor * prev_right_line[2] + (1-smooth_factor) * right_x_end)
    else:
        # Use previous line if available, otherwise default
        if prev_right_line is not None:
            right_x_start, _, right_x_end, _ = prev_right_line
        else:
            right_x_start, right_x_end = width, width

    # Save current lines for next frame
    if left_x_start != 0 and left_x_end != 0:
        prev_left_line = [left_x_start, max_y, left_x_end, min_y]
    if right_x_start != width and right_x_end != width:
        prev_right_line = [right_x_start, max_y, right_x_end, min_y]

    # Create the filled polygon between the left and right lane lines
    lane_image = draw_lane_lines(
        image,
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y]
    )

    return lane_image

# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width, bbox_height):
    # For simplicity, assume the distance is inversely proportional to the box size
    # This is a basic estimation, you may use camera calibration for more accuracy
    focal_length = 1000  # Example focal length, modify based on camera setup
    known_width = 2.0  # Approximate width of the car (in meters)
    distance = (known_width * focal_length) / bbox_width  # Basic distance estimation
    return distance

# Main function to read and process video with YOLOv8
def process_video():
    # Open the video file
    cap = cv2.VideoCapture('video/car.mp4')

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Set the desired frame rate
    target_fps = 30
    frame_time = 1.0 / target_fps  # Time per frame to maintain 30fps

    # Resize to 720p (1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Optional: create video writer for saving the processed video
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, target_fps, (1280, 720))

    # Loop through each frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame to 720p
        resized_frame = cv2.resize(frame, (1280, 720))

        # Run the lane detection pipeline
        lane_frame = pipeline(resized_frame)

        # Optional: write the frame to output video
        # out.write(lane_frame)

        # Display the resulting frame with both lane detection and car detection
        cv2.imshow('Lane and Car Detection', lane_frame)

        # Limit the frame rate to 30fps
        time.sleep(frame_time)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    # out.release()  # Uncomment if using VideoWriter
    cv2.destroyAllWindows()

# Run the video processing function
if __name__ == "__main__":
    process_video()