import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def detect_lines(image):
    lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    return lines

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def process_image(image_path):
    image = load_image(image_path)
    lane_image = np.copy(image)
    edges = preprocess_image(lane_image)
    cropped_edges = region_of_interest(edges)
    lines = detect_lines(cropped_edges)
    line_image = display_lines(lane_image, lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    cv2.imshow('Lane Detection', combo_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        edges = preprocess_image(frame)
        cropped_edges = region_of_interest(edges)
        lines = detect_lines(cropped_edges)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        
        cv2.imshow('Lane Detection Video', combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# process_image("test/test_image.jpg")
process_video("video/sr.mp4")
