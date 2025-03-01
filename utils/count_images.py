import os
import yaml
from collections import defaultdict

def load_yaml(yaml_path):
    # Convert to absolute path
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def count_images(yaml_path):
    data = load_yaml(yaml_path)
    class_counts = defaultdict(int)
    
    # Get the base directory from yaml file location
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    
    for split in ['train', 'val', 'test']:
        # Construct absolute paths for images and labels
        image_dir = os.path.abspath(os.path.join(base_dir, data[split]))
        label_dir = os.path.abspath(image_dir.replace('images', 'labels'))
        
        # Print paths for debugging
        print(f"Image directory for {split}: {image_dir}")
        print(f"Label directory for {split}: {label_dir}")
        
        # Check if directory exists
        if not os.path.exists(label_dir):
            print(f"Warning: Directory not found: {label_dir}")
            continue
            
        for label_file in os.listdir(label_dir):
            with open(os.path.join(label_dir, label_file), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    class_name = data['names'][class_id]
                    class_counts[class_name] += 1
    
    return class_counts

if __name__ == "__main__":
    # Use absolute path for yaml file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(os.path.dirname(current_dir), 'merged_data', 'data.yaml')
    
    class_counts = count_images(yaml_path)
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
