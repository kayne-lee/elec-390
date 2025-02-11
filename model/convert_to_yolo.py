import os
import xml.etree.ElementTree as ET
import cv2

# Directory paths model/example_dataset/annotations
images_dir = 'example_dataset/images'  # Replace with the path to your images directory
annotations_dir = 'example_dataset/annotations'  # Replace with the path to your annotations directory
output_dir = 'example_dataset/yolo_annotations'  # The directory to save YOLO annotations

# Mapping of class names to class IDs (Modify as per your dataset)
class_names = {
    'sign_stop': 0,  # Add more class names here with unique IDs
    'sign_oneway_right': 1,
    'sign_oneway_left': 2,
    'sign_noentry': 3,
    'sign_yield': 4,
    'road_crosswalk': 5,
    'road_oneway': 6,
    'vehicle': 7,
    'duck_regular': 8,
    'duck_specialty': 9
    
}

# Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)

def convert_annotation(xml_file, image_width, image_height):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Open the corresponding YOLO annotation text file
    yolo_txt_file = os.path.join(output_dir, os.path.basename(xml_file).replace('.xml', '.txt'))
    with open(yolo_txt_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_names:
                continue  # Skip if the class name is not in the mapping
            
            # Get the bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / float(image_width)
            height = (ymax - ymin) / float(image_height)

            # Get the class ID
            class_id = class_names[class_name]

            # Write to the text file in YOLO format
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def main():
    # Loop through each annotation file
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue

        # Full path to the XML file
        xml_file_path = os.path.join(annotations_dir, xml_file)

        # Get the corresponding image file name (without extension)
        image_file_name = xml_file.replace('.xml', '.jpg')
        image_file_path = os.path.join(images_dir, image_file_name)

        # Check if the image exists
        if not os.path.exists(image_file_path):
            print(f"Warning: Image for {xml_file} not found.")
            continue

        # Read the image to get its width and height
        image = cv2.imread(image_file_path)
        image_height, image_width, _ = image.shape

        # Convert the annotation to YOLO format
        convert_annotation(xml_file_path, image_width, image_height)

if __name__ == '__main__':
    main()
