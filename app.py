import os
import io
import cv2 as cv
import uuid
import shutil
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from datetime import datetime
import zipfile
from pathlib import Path
import json

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['PROJECTS_FOLDER'] = 'projects'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Ensure projects directory exists
os.makedirs(app.config['PROJECTS_FOLDER'], exist_ok=True)

# Load YOLO model
try:
    model_path = '/home/bheeshmsharma/Sabari/new_tool-copy/models/defect_runs/exp_defect/weights/best.pt'
    model = YOLO(model_path)
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
           
import cv2 as cv
import numpy as np

import cv2 as cv
import numpy as np

import cv2 as cv
import numpy as np

def preprocess_img(img_path, min_contour_area_ratio=0.01, pad=10):
    try:
        # Load image
        original = cv.imread(img_path)
        if original is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # Convert to grayscale
        gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

        # Threshold to binary
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Find contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv.cvtColor(original, cv.COLOR_BGR2RGB)

        # Image metadata
        img_h, img_w = original.shape[:2]
        img_area = img_w * img_h

        # Select largest contour above min area ratio
        largest_contour = max(
            (cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area_ratio * img_area),
            key=cv.contourArea,
            default=None
        )
        
        if largest_contour is None:
            return cv.cvtColor(original, cv.COLOR_BGR2RGB)

        # Create black mask and draw the largest contour filled
        mask = np.zeros_like(gray)
        cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)

        # Apply mask to keep only the object
        result = cv.bitwise_and(original, original, mask=mask)

        # Bounding box of largest contour
        x, y, w, h = cv.boundingRect(largest_contour)

        # Add padding, making sure not to go outside image
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        cropped = result[y1:y2, x1:x2]

        return cv.cvtColor(cropped, cv.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return cv.cvtColor(original, cv.COLOR_BGR2RGB)



def get_project_paths(project_name):
    base = os.path.join(app.config['PROJECTS_FOLDER'], secure_filename(project_name))
    # Updated folder structure
    return {
        'base': base,
        'original': {
            'base': os.path.join(base, 'original'),
            'images': os.path.join(base, 'original', 'images'),
            'labels': os.path.join(base, 'original', 'labels')
        },
        'additions': {
            'base': os.path.join(base, 'additions'),
            'images': os.path.join(base, 'additions', 'images'),
            'labels': os.path.join(base, 'additions', 'labels')
        },
        'modified': {
            'base': os.path.join(base, 'modified'),
            'images': os.path.join(base, 'modified', 'images'),
            'labels': os.path.join(base, 'modified', 'labels')
        },
        'config': os.path.join(base, 'project.json')
    }

@app.route('/')
def index():
    # List existing projects
    projects = []
    if os.path.exists(app.config['PROJECTS_FOLDER']):
        for name in os.listdir(app.config['PROJECTS_FOLDER']):
            project_path = os.path.join(app.config['PROJECTS_FOLDER'], name)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, 'project.json')
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        projects.append(json.load(f))
    return render_template('index.html', projects=projects)

@app.route('/create_project', methods=['POST'])
def create_project():
    project_name = request.form['project_name']
    paths = get_project_paths(project_name)
    
    # Create project directories with new structure
    os.makedirs(paths['base'], exist_ok=True)
    os.makedirs(paths['original']['base'], exist_ok=True)
    os.makedirs(paths['original']['images'], exist_ok=True)
    os.makedirs(paths['original']['labels'], exist_ok=True)
    os.makedirs(paths['additions']['base'], exist_ok=True)
    os.makedirs(paths['additions']['images'], exist_ok=True)
    os.makedirs(paths['additions']['labels'], exist_ok=True)
    os.makedirs(paths['modified']['base'], exist_ok=True)
    os.makedirs(paths['modified']['images'], exist_ok=True)
    os.makedirs(paths['modified']['labels'], exist_ok=True)
    
    # Create project config
    config = {
        'name': project_name,
        'created': str(datetime.now()),
        'image_count': 0
    }
    with open(paths['config'], 'w') as f:
        json.dump(config, f)
    
    return jsonify({'status': 'success', 'project_name': project_name})

@app.route('/upload/<project_name>', methods=['POST'])
def upload_folder(project_name):
    paths = get_project_paths(project_name)
    
    # Save uploaded files to original images folder
    uploaded_files = []
    for file in request.files.getlist('files'):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_path = os.path.join(paths['original']['images'], filename)
            file.save(original_path)
            uploaded_files.append(filename)
    
    # Run YOLO inference on uploaded images
    process_images(project_name, uploaded_files)
    
    # Update project config
    with open(paths['config'], 'r+') as f:
        config = json.load(f)
        config['image_count'] = len(os.listdir(paths['original']['images']))
        f.seek(0)
        json.dump(config, f)
        f.truncate()
    
    return redirect(url_for('annotate', project_name=project_name))

def process_images(project_name, image_files=None):
    """Process images with YOLO and save annotations"""
    paths = get_project_paths(project_name)
    
    if image_files is None:
        image_files = [f for f in os.listdir(paths['original']['images']) if allowed_file(f)]
    
    for filename in image_files:
        orig_img_path = os.path.join(paths['original']['images'], filename)
        label_path = os.path.join(paths['original']['labels'], os.path.splitext(filename)[0] + '.txt')
        
        # Preprocess and overwrite the original image
        preprocessed_img = preprocess_img(orig_img_path)
        # Save preprocessed image as the main image
        cv.imwrite(orig_img_path, cv.cvtColor(preprocessed_img, cv.COLOR_RGB2BGR))
        
        # Only process if annotation doesn't exist
        if not os.path.exists(label_path) and model is not None:
            try:
                results = model(orig_img_path)
                with open(label_path, 'w') as f:
                    boxes = results[0].boxes
                    if boxes is not None and boxes.xywhn is not None:
                        for i in range(len(boxes)):
                            class_id = int(boxes.cls[i].item())
                            x_center, y_center, width, height = boxes.xywhn[i].tolist()
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            except Exception as e:
                print(f"Error in YOLO processing: {e}")
        
        # Copy preprocessed image to additions and modified folders
        additions_img_path = os.path.join(paths['additions']['images'], filename)
        modified_img_path = os.path.join(paths['modified']['images'], filename)
        shutil.copy2(orig_img_path, additions_img_path)
        shutil.copy2(orig_img_path, modified_img_path)
@app.route('/annotate')
def annotate():
    project_name = request.args.get('project_name')
    paths = get_project_paths(project_name)
    images = []
    if os.path.exists(paths['original']['images']):
        images = [f for f in os.listdir(paths['original']['images']) if allowed_file(f)]
    return render_template('annotate.html', project_name=project_name, images=images)

@app.route('/get_image/<project_name>/<filename>')
def get_image(project_name, filename):
    paths = get_project_paths(project_name)
    image_path = os.path.join(paths['original']['images'], filename)
    if not os.path.exists(image_path):
        return "Image not found", 404
    return send_file(image_path)

@app.route('/get_annotations/<project_name>/<image_name>/<view_mode>')
def get_annotations(project_name, image_name, view_mode):
    paths = get_project_paths(project_name)
    base_filename = os.path.splitext(image_name)[0]
    label_filename = base_filename + '.txt'
    
    annotations = []
    deleted_annotations = []
    
    # Helper function to read annotations
    def read_annotations(path, is_original=False):
        annotations_from_file = []
        deleted_from_file = []
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):  # Handle commented out (deleted) annotations
                        if is_original:  # Only process deletions from original
                            parts = line[1:].strip().split()
                            if len(parts) == 5:
                                class_id, x_center, y_center, width, height = map(float, parts)
                                deleted_from_file.append({
                                    'class': int(class_id),
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height,
                                    'is_original': True,
                                    'deleted': True
                                })
                    else:  # Regular annotations
                        parts = line.split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            annotations_from_file.append({
                                'class': int(class_id),
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height,
                                'is_original': is_original,  # Tag as original or not
                                'deleted': False
                            })
        return annotations_from_file, deleted_from_file
    
    # --- Custom logic for view modes ---
    if view_mode == 'additions':
        # Show user-added annotations and non-deleted original annotations as reference
        additions_path = os.path.join(paths['additions']['labels'], label_filename)
        original_path = os.path.join(paths['original']['labels'], label_filename)
        modified_path = os.path.join(paths['modified']['labels'], label_filename)
        # User-added
        additions_annotations, _ = read_annotations(additions_path)
        annotations.extend([ann for ann in additions_annotations if not ann.get('is_original', False)])
        # Reference: non-deleted original
        original_annotations, _ = read_annotations(original_path, True)
        # Build a set of deleted boxes from modified/labels (commented lines)
        deleted_set = set()
        if os.path.exists(modified_path):
            with open(modified_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        parts = line[1:].strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            deleted_set.add((int(class_id), round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)))
        for ann in original_annotations:
            key = (ann['class'], round(ann['x_center'], 6), round(ann['y_center'], 6), round(ann['width'], 6), round(ann['height'], 6))
            if key not in deleted_set:
                ann['is_reference'] = True
                annotations.append(ann)
        # No deleted_annotations in this view
        return jsonify({
            'annotations': annotations,
            'deleted_annotations': []
        })
    
    if view_mode == 'modified':
        # Show all original annotations, mark as deleted if present in modified/labels as commented lines
        original_path = os.path.join(paths['original']['labels'], label_filename)
        modified_path = os.path.join(paths['modified']['labels'], label_filename)
        original_annotations, _ = read_annotations(original_path, True)
        # Build a set of deleted boxes from modified/labels (commented lines)
        deleted_set = set()
        if os.path.exists(modified_path):
            with open(modified_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        parts = line[1:].strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            deleted_set.add((int(class_id), round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)))
        # Mark original annotations as deleted if present in deleted_set
        for ann in original_annotations:
            key = (ann['class'], round(ann['x_center'], 6), round(ann['y_center'], 6), round(ann['width'], 6), round(ann['height'], 6))
            if key in deleted_set:
                ann['deleted'] = True
                deleted_annotations.append(ann)
            else:
                ann['deleted'] = False
                annotations.append(ann)
        return jsonify({
            'annotations': annotations,
            'deleted_annotations': deleted_annotations
        })

    if view_mode == 'all':
        # Show user-added, non-deleted original, and deleted original (dotted) annotations
        original_path = os.path.join(paths['original']['labels'], label_filename)
        additions_path = os.path.join(paths['additions']['labels'], label_filename)
        modified_path = os.path.join(paths['modified']['labels'], label_filename)
        # Load all user-added
        additions_annotations, _ = read_annotations(additions_path)
        annotations.extend([ann for ann in additions_annotations if not ann.get('is_original', False)])
        # Load all original
        original_annotations, _ = read_annotations(original_path, True)
        # Build a set of deleted boxes from modified/labels (commented lines)
        deleted_set = set()
        if os.path.exists(modified_path):
            with open(modified_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        parts = line[1:].strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            deleted_set.add((int(class_id), round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)))
        # Add non-deleted original annotations
        for ann in original_annotations:
            key = (ann['class'], round(ann['x_center'], 6), round(ann['y_center'], 6), round(ann['width'], 6), round(ann['height'], 6))
            if key not in deleted_set:
                ann['deleted'] = False
                annotations.append(ann)
            else:
                ann['deleted'] = True
                deleted_annotations.append(ann)
        return jsonify({
            'annotations': annotations,
            'deleted_annotations': deleted_annotations
        })
    
    # --- Existing logic for other modes ---
    # Load annotations based on view mode
    if view_mode == 'original' or view_mode == 'all':
        original_path = os.path.join(paths['original']['labels'], label_filename)
        original_annotations, original_deleted = read_annotations(original_path, True)
        annotations.extend(original_annotations)
        deleted_annotations.extend(original_deleted)
    
    if view_mode == 'additions' or view_mode == 'all':
        additions_path = os.path.join(paths['additions']['labels'], label_filename)
        additions_annotations, _ = read_annotations(additions_path)
        # Filter to only include non-original annotations (additions)
        added = [ann for ann in additions_annotations if not ann.get('is_original', False)]
        annotations.extend(added)
    
    if view_mode == 'modified' or view_mode == 'all':
        modified_path = os.path.join(paths['modified']['labels'], label_filename)
        modified_annotations, modified_deleted = read_annotations(modified_path)
        # Filter to only include modified annotations
        annotations.extend(modified_annotations)
        deleted_annotations.extend(modified_deleted)
    
    return jsonify({
        'annotations': annotations,
        'deleted_annotations': deleted_annotations
    })

@app.route('/save_annotations/<project_name>/<image_name>/<view_mode>', methods=['POST'])
def save_annotations(project_name, image_name, view_mode):
    paths = get_project_paths(project_name)
    base_filename = os.path.splitext(image_name)[0]
    label_filename = base_filename + '.txt'
    
    try:
        data = request.json
        annotations = data.get('annotations', [])
        deleted_annotations = data.get('deleted_annotations', [])
        
        original_label_path = os.path.join(paths['original']['labels'], label_filename)
        additions_label_path = os.path.join(paths['additions']['labels'], label_filename)
        modified_label_path = os.path.join(paths['modified']['labels'], label_filename)
        
        # Read original annotations
        original_annotations = []
        if os.path.exists(original_label_path):
            with open(original_label_path, 'r') as f:
                original_annotations = [line.strip() for line in f if line.strip()]
        
        # Split annotations
        original_boxes = [ann for ann in annotations if ann.get('is_original', False)]
        added_boxes = [ann for ann in annotations if not ann.get('is_original', False)]
        
        # Save only user-added boxes in additions/labels if in 'additions' view
        if view_mode == 'additions':
            if added_boxes:
                with open(additions_label_path, 'w') as f:
                    for ann in added_boxes:
                        class_id = int(ann['class'])
                        x_center = float(ann['x_center'])
                        y_center = float(ann['y_center'])
                        width = float(ann['width'])
                        height = float(ann['height'])
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            # Do not write to modified/labels
            return jsonify({'status': 'success'})
        
        # Save only deleted original boxes in modified/labels if in 'modified' view
        if view_mode == 'modified':
            if deleted_annotations:
                # Only write deleted original boxes as commented lines
                with open(modified_label_path, 'w') as f:
                    for deleted in deleted_annotations:
                        if deleted.get('is_original', False):
                            f.write(f"# {int(deleted['class'])} {deleted['x_center']:.6f} {deleted['y_center']:.6f} {deleted['width']:.6f} {deleted['height']:.6f}\n")
            # Do not write user-added boxes here
            return jsonify({'status': 'success'})
        
        # In other view modes, do not write to additions/labels or modified/labels
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error saving annotations: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/auto_annotate/<project_name>/<image_name>', methods=['POST'])
def auto_annotate(project_name, image_name):
    """Run YOLO detection on a single image"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'YOLO model not available'})
    
    try:
        paths = get_project_paths(project_name)
        img_path = os.path.join(paths['original']['images'], image_name)
        label_path = os.path.join(paths['original']['labels'], os.path.splitext(image_name)[0] + '.txt')
        
        # Run YOLO on the image
        results = model(img_path)
        
        # Save the results to the labels file
        with open(label_path, 'w') as f:
            boxes = results[0].boxes
            if boxes is not None and boxes.xywhn is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    x_center, y_center, width, height = boxes.xywhn[i].tolist()
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Ensure the image is copied to additions and modified directories
        additions_img_path = os.path.join(paths['additions']['images'], image_name)
        modified_img_path = os.path.join(paths['modified']['images'], image_name)
        
        if not os.path.exists(additions_img_path):
            shutil.copy2(img_path, additions_img_path)
        
        if not os.path.exists(modified_img_path):
            shutil.copy2(img_path, modified_img_path)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error in auto annotation: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/export_dataset/<project_name>')
def export_dataset(project_name):
    paths = get_project_paths(project_name)

    # Create an in-memory zip file
    mem_zip = io.BytesIO()

    try:
        with zipfile.ZipFile(mem_zip, 'w') as zipf:
            # Add images
            for img in os.listdir(paths['original']['images']):
                if allowed_file(img):
                    img_path = os.path.join(paths['original']['images'], img)
                    zipf.write(img_path, arcname=f"images/{img}")

            # Gather all label filenames from original and additions
            label_filenames = set()
            if os.path.exists(paths['original']['labels']):
                label_filenames.update([f for f in os.listdir(paths['original']['labels']) if f.endswith('.txt')])
            if os.path.exists(paths['additions']['labels']):
                label_filenames.update([f for f in os.listdir(paths['additions']['labels']) if f.endswith('.txt')])

            # For each label file, merge additions and (original - modified)
            for lbl in label_filenames:
                merged_lines = []
                # 1. Add all lines from additions/labels (if exists)
                additions_path = os.path.join(paths['additions']['labels'], lbl)
                if os.path.exists(additions_path):
                    with open(additions_path, 'r') as f:
                        merged_lines.extend([line for line in f if line.strip() and not line.strip().startswith('#')])

                # 2. Add lines from original/labels that are not commented out in modified/labels
                original_path = os.path.join(paths['original']['labels'], lbl)
                modified_path = os.path.join(paths['modified']['labels'], lbl)
                original_lines = []
                if os.path.exists(original_path):
                    with open(original_path, 'r') as f:
                        original_lines = [line for line in f if line.strip() and not line.strip().startswith('#')]

                # Build set of deleted lines from modified/labels (commented lines, stripped)
                deleted_set = set()
                if os.path.exists(modified_path):
                    with open(modified_path, 'r') as f:
                        for line in f:
                            if line.strip().startswith('#'):
                                deleted_set.add(line[1:].strip())

                # Add only those original lines not deleted in modified (strip for comparison)
                for line in original_lines:
                    if line.strip() not in {d.strip() for d in deleted_set}:
                        merged_lines.append(line)

                # Write merged label file to zip (if any lines)
                if merged_lines:
                    #print(f"[EXPORT DEBUG] {lbl} merged lines:\n{''.join(merged_lines)}")
                    zipf.writestr(f"labels/{lbl}", ''.join(merged_lines))

            # Add data.yaml
            yaml_content = """train: images/
val: images/
nc: 1
names: ['defect']
"""
            zipf.writestr('data.yaml', yaml_content)

        # Prepare in-memory zip for sending
        mem_zip.seek(0)
        return send_file(
            mem_zip,
            mimetype='application/zip',
            download_name=f"{project_name}_dataset.zip",
            as_attachment=True
        )
    except Exception as e:
        print(f"Error exporting dataset: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)