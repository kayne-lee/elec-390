import os
import cv2
import pandas as pd

# ----------------------------------------------------
# Configuration: Update these paths as needed
# ----------------------------------------------------
# Root directory of your dataset (where Train/, train.csv, etc. are located)
dataset_dir = "archive 2"  # e.g., "/home/user/datasets/my_dataset"

# Path to the training CSV file
train_csv_path = os.path.join(dataset_dir, "Train.csv")

# Folder where training images are located.
# Here we assume that the "Path" column in train.csv is relative to dataset_dir.
# For example, "Train/20/00020_00000_00000.png"
# If your images are stored in a different folder, update this accordingly.
# In many cases, the images may be directly referenced by the CSV "Path" column.
# We'll join them with dataset_dir.
# If images are in a separate folder (e.g., "images"), update images_dir below.
images_dir = dataset_dir

# Output folder for YOLO-format label files
labels_dir = os.path.join(dataset_dir, "labels")
os.makedirs(labels_dir, exist_ok=True)

# ----------------------------------------------------
# Read train.csv and process annotations for every image
# ----------------------------------------------------
print("Reading training annotations from:", train_csv_path)
try:
    train_df = pd.read_csv(train_csv_path)
except Exception as e:
    raise Exception(f"Error reading train.csv at {train_csv_path}: {e}")

# Expected columns:
#   "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"
expected_cols = {"Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"}
if not expected_cols.issubset(set(train_df.columns)):
    raise Exception(f"train.csv is missing one or more expected columns: {expected_cols}")

# Group annotations by image path (each image may have multiple annotations)
grouped = train_df.groupby("Path")

for rel_path, group in grouped:
    # Construct the full path to the image.
    image_path = os.path.join(images_dir, rel_path)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image '{image_path}' not found. Skipping...")
        continue

    # Option 1: Use the provided image dimensions from the CSV.
    # (Assumes that all rows for this image have the same Width and Height.)
    try:
        img_width = float(group.iloc[0]["Width"])
        img_height = float(group.iloc[0]["Height"])
    except Exception as e:
        print(f"Warning: Could not obtain image dimensions for '{image_path}' from CSV: {e}")
        continue

    # Option 2 (alternative): Load the image to get its actual dimensions.
    # Uncomment the lines below if you prefer to use the real image dimensions.
    # image = cv2.imread(image_path)
    # if image is None:
    #     print(f"Warning: Could not load image '{image_path}'. Skipping...")
    #     continue
    # img_height, img_width = image.shape[:2]

    yolo_lines = []  # List to store YOLO annotation lines for this image

    for idx, row in group.iterrows():
        try:
            class_id = int(row["ClassId"])
        except Exception as e:
            print(f"Warning: Invalid ClassId in row {idx} for image '{rel_path}': {e}")
            continue

        try:
            # Retrieve bounding box coordinates (in pixels)
            roi_x1 = float(row["Roi.X1"])
            roi_y1 = float(row["Roi.Y1"])
            roi_x2 = float(row["Roi.X2"])
            roi_y2 = float(row["Roi.Y2"])
        except Exception as e:
            print(f"Warning: Error reading ROI coordinates in row {idx} for image '{rel_path}': {e}")
            continue

        # Convert bounding box coordinates to YOLO format (normalized)
        x_center = ((roi_x1 + roi_x2) / 2.0) / img_width
        y_center = ((roi_y1 + roi_y2) / 2.0) / img_height
        box_width = (roi_x2 - roi_x1) / img_width
        box_height = (roi_y2 - roi_y1) / img_height

        # Format: <class_id> <x_center> <y_center> <box_width> <box_height>
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        yolo_lines.append(line)

    # Create a label file for this image only if at least one annotation was processed.
    if yolo_lines:
        # Use the image file's base name (without extension) for the label file.
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        label_filename = base_name + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        print(f"Created label file for '{rel_path}' with {len(yolo_lines)} annotation(s).")
    else:
        print(f"No annotations processed for '{rel_path}'. Skipping label file creation.")
# import os
# import shutil

# # Define the Train directory
# train_dir = "archive 2/Train"

# # Loop through each numbered folder (0-42)
# for i in range(43):  # 0 to 42 inclusive
#     subfolder_path = os.path.join(train_dir, str(i))
#     print(subfolder_path)
    
#     # Check if the subfolder exists
#     if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
#         # Move all files from the subfolder to Train directory
#         for file in os.listdir(subfolder_path):
#             file_path = os.path.join(subfolder_path, file)
#             if os.path.isfile(file_path):  # Ensure it's a file
#                 shutil.move(file_path, train_dir)
        
#         # Remove the now-empty folder
#         os.rmdir(subfolder_path)
#     else:
#         print(f"Folder '{subfolder_path}' not found. Skipping...")

# print("All files moved and folders deleted successfully!")
