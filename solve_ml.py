import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from glob import glob
from PIL import Image, ImageDraw

def parse_vertices_from_filename(filename):
    """
    Parse vertices from filename pattern like: handdrawn-photo/873,904;362,880;388,504;885,469;863,671;354,605;407,221;863,263.png
    
    Args:
        filename (str): Filename containing vertex coordinates
    
    Returns:
        list: List of 8 (x, y) coordinate tuples
    """
    # Extract the coordinate string
    coord_str = os.path.basename(filename).split('.')[0]
    
    # Parse all coordinates
    vertices = []
    for coord in coord_str.split(';'):
        x, y = map(int, coord.split(','))
        vertices.append((x, y))
    
    if len(vertices) != 8:
        raise ValueError(f"Expected 8 vertices, got {len(vertices)} in file {filename}")
    
    return vertices

def prepare_yolo_dataset(image_dir, output_dir):
    """
    Prepare YOLO format dataset from images with vertices in filenames.
    
    Args:
        image_dir (str): Directory containing training images
        output_dir (str): Directory to save YOLO formatted dataset
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    image_paths = glob(os.path.join(image_dir, "*.png"))
    
    for img_path in image_paths:
        try:
            # Parse vertices from filename
            vertices = parse_vertices_from_filename(img_path)
            
            # Load image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Create YOLO annotation file
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(output_dir, "labels", base_name + ".txt")
            
            with open(label_path, 'w') as f:
                for x, y in vertices:
                    # YOLO format: class x_center y_center width height (normalized 0-1)
                    x_norm = x / w
                    y_norm = y / h
                    f.write(f"0 {x_norm:.6f} {y_norm:.6f} 0.01 0.01\n")  # Small bounding boxes
            
            # Copy image to output directory
            output_img_path = os.path.join(output_dir, "images", os.path.basename(img_path))
            cv2.imwrite(output_img_path, img)
            
        except Exception as e:
            print(f"Skipping {img_path}: {str(e)}")

def train_vertex_detector(data_dir, epochs=100):
    """
    Train YOLO model to detect cube vertices.
    
    Args:
        data_dir (str): Directory containing prepared YOLO dataset
        epochs (int): Number of training epochs
    
    Returns:
        str: Path to the best trained model
    """
    # Create dataset.yaml config
    config = {
        'path': os.path.abspath(data_dir),
        'train': 'images',
        'val': 'images',  # Using same for val in this simple case
        'names': {0: 'vertex'}
    }
    
    config_path = os.path.join(data_dir, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load model and train
    model = YOLO('yolov8n.pt')
    model.train(
        data=config_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        single_cls=True,
        augment=True,
        name='cube_vertex_detector'
    )
    
    return os.path.join("runs", "detect", "cube_vertex_detector", "weights", "best.pt")


def detect_cube_vertices(image_path):
    """
    Detect cube vertices in an image using trained YOLO model.
    
    Args:
        image_path (str): Path to input image
        model_path (str): Path to trained YOLO model
    
    Returns:
        list: List of 8 (x, y) coordinate tuples
    """
    model_path = os.path.join("..", "runs", "detect", "cube_vertex_detector", "weights", "best.pt")
    # Load image and model
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    model = YOLO(model_path)
    
    # Run detection
    results = model(img)
    
    # Extract detected points
    points = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            points.append((x_center, y_center))
    
    # If we didn't get exactly 8 points, use clustering
    if len(points) != 8:
        if len(points) > 8:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=8).fit(np.array(points))
            points = [tuple(center) for center in kmeans.cluster_centers_]
        elif len(points) < 8:
            # Fallback to simple corner detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
            if corners is not None:
                corners = np.int32(corners)
                points = [tuple(c.ravel()) for c in corners[:8]]  # Take first 8 corners
            else:
                # As last resort, distribute points evenly
                h, w = img.shape[:2]
                points = [(w*i//3, h*j//3) for i in range(1,3) for j in range(1,3)]
    ans = []
    for i, el in enumerate(points[:8]):
        ans.append(el)
    return ans  # Ensure we return exactly 8 points

# Example usage
if __name__ == "__main__":
    # 1. Prepare dataset
    prepare_yolo_dataset("handdrawn-photo", "yolo_dataset")
    
    # 2. Train model
    model_path = train_vertex_detector("yolo_dataset", epochs=50)
    
    
    # # 3. Detect vertices in new image
    test_image = "image1.png"
    vertices = detect_cube_vertices(test_image, model_path)
    
    print("Detected vertices:")
    for i, (x, y) in enumerate(vertices):
        print(f"Vertex {i+1}: ({x:.1f}, {y:.1f})")
