import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load your trained YOLO model (change the path to your model)
model = YOLO('best.pt')

# Define the directory containing the images to be checked
image_dir = 'Screenshot 2025-02-04 at 4.10.45 PM.png'

# Function to process a single image and show results
def check_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Make predictions on the image
    results = model(img)  # Run the model on the image
    
    # Print results
    print(f"Results for {image_path}: {results}")

    # Draw bounding boxes on the image
    annotated_img = results[0].plot()  # Annotate the image with detected boxes

    # Display the image with matplotlib
    plt.imshow(annotated_img)
    plt.title(f"Predictions for {os.path.basename(image_path)}")
    plt.axis('off')  # Hide axis
    plt.show()

# If you want to check multiple images in a directory
def check_images_in_directory(image_dir):
    # Iterate through each image in the directory
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, file)
            check_image(image_path)  # Process and display each image

# Call the function to process images in the directory
check_images_in_directory(image_dir)
