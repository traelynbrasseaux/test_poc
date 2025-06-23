from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8 model (coco dataset)
yolo_model = YOLO('yolov8x.pt')  # Use the nano model for speed; can change up to yolov8x.pt for more accuracy

# Confidence threshold for detections (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.15  # Default YOLOv8 threshold, can be lowered to catch more objects

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Improved cleaning instructions for common and edge-case objects
CLEANING_INSTRUCTIONS = {
    # Kitchenware
    'sink': 'Scrub the sink and ensure it is free of stains or debris.',
    'cup': "Wash the cup and put it away. If the item doesn't belong to the property or appears valuable/personal (e.g., a travel mug), set it aside and alert the property manager.",
    'bowl': 'Wash the bowl and put it away.',
    'bottle': "Wash or recycle the bottle, and wipe the area if needed. If the bottle is reusable or appears valuable/personal, set it aside and alert the property manager.",
    'wine glass': "Wash the wine glass and put it away. If the item doesn't belong to the property or appears valuable/personal, set it aside and alert the property manager.",
    'fork': 'Wash the fork and put it away.',
    'knife': 'Wash the knife and put it away.',
    'spoon': 'Wash the spoon and put it away.',
    # Food items
    'banana': 'Dispose of banana peels and wipe the area.',
    'apple': 'Dispose of apple cores and wipe the area.',
    'sandwich': 'Dispose of leftovers and wipe the area.',
    'orange': 'Dispose of peels and wipe the area.',
    'broccoli': 'Dispose of leftovers and wipe the area.',
    'carrot': 'Dispose of leftovers and wipe the area.',
    'hot dog': 'Dispose of leftovers and wipe the area.',
    'pizza': 'Dispose of leftovers and wipe the area.',
    'donut': 'Dispose of leftovers and wipe the area.',
    'cake': 'Dispose of leftovers and wipe the area.',
    # Furniture
    'chair': 'Wipe the chair if needed and check for stains. Arrange neatly.',
    'couch': 'Tidy the couch and arrange any pillows or blankets neatly. Spot clean if there are visible stains.',
    'bed': 'Make the bed neatly and arrange pillows and blankets.',
    'dining table': 'Wipe the table and remove any debris.',
    'table': 'Wipe the table and remove any debris.',
    'bench': 'Wipe the bench if needed and check for stains. Arrange neatly.',
    # Soft furnishings
    'pillow': 'Arrange the pillow neatly. Spot clean if needed.',
    'blanket': 'Fold and arrange the blanket neatly.',
    'teddy bear': "Arrange the teddy bear neatly. If it appears valuable or personal, set it aside and alert the property manager. Spot clean if needed.",
    # Plants
    'potted plant': 'Ensure the plant is upright and the area is tidy. Water if needed.',
    # Bathroom
    'toilet': 'Clean and disinfect the toilet.',
    'toothbrush': 'Store toothbrushes properly. Dispose if left behind.',
    'hair drier': 'Store the hair drier properly. If it appears valuable or personal, set it aside and alert the property manager.',
    # Electronics
    'tv': 'Dust the TV and wipe the screen gently with a dry cloth. Do not use water.',
    'laptop': 'Wipe the laptop gently with a dry cloth. If left behind, set it aside and alert the property manager.',
    'remote': 'Wipe the remote gently and place it in an obvious spot. If it does not belong to the property, alert the property manager.',
    'keyboard': 'Wipe the keyboard gently with a dry cloth. If left behind, set it aside and alert the property manager.',
    'cell phone': 'Remove and store any left-behind phones. Alert the property manager.',
    # Kitchen appliances
    'microwave': 'Wipe the inside and outside of the microwave.',
    'oven': 'Wipe the inside and outside of the oven.',
    'toaster': 'Wipe the toaster and remove crumbs.',
    'refrigerator': 'Wipe the refrigerator and check for leftover food.',
    # Decor and misc
    'clock': 'Dust the clock and ensure it is visible. If it appears valuable or personal, set it aside and alert the property manager.',
    'vase': 'Ensure the vase is upright and clean. If it appears valuable or personal, set it aside and alert the property manager.',
    'scissors': 'Store scissors safely. If they appear valuable or personal, set them aside and alert the property manager.',
    # Sports and toys
    'sports ball': 'Store the sports ball neatly. If it appears valuable or personal, set it aside and alert the property manager.',
    'skateboard': 'Store the skateboard neatly. If it appears valuable or personal, set it aside and alert the property manager.',
    'frisbee': 'Store the frisbee neatly. If it appears valuable or personal, set it aside and alert the property manager.',
    # Books and paper
    'book': 'Dust the book and arrange it neatly. If it appears valuable or personal, set it aside and alert the property manager.',
    # Countertops (not a COCO class, but for custom extension)
    'counter': 'Wipe the countertop and remove any debris.',
    'countertop': 'Wipe the countertop and remove any debris.',
    # Add more as needed for your property types
}
# For any object not in the mapping, provide a generic fallback:
# "Clean and tidy the [object]. If unsure, refer to property guidelines."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def yolo_detect(image_path, confidence_threshold=None):
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
    
    results = yolo_model(image_path, conf=confidence_threshold)[0]
    detections = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        detections.append({
            'label': label,
            'confidence': conf,
            'bbox': xyxy.tolist()  # [x1, y1, x2, y2]
        })
    
    return detections

def draw_boxes(image_path, detections, out_path):
    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(out_path, img)

def remove_duplicate_detections(detections, iou_threshold=0.5):
    filtered = []
    used = set()
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        keep = True
        for j, det2 in enumerate(detections):
            if i == j or det1['label'] != det2['label']:
                continue
            # Calculate IoU
            box1 = det1['bbox']
            box2 = det2['bbox']
            xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
            xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > iou_threshold and det1['confidence'] < det2['confidence']:
                keep = False
                break
        if keep:
            filtered.append(det1)
            used.add(i)
    return filtered

def suppress_overlapping_classes(detections, iou_threshold=0.8):
    filtered = []
    used = set()
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        keep = True
        for j, det2 in enumerate(detections):
            if i == j:
                continue
            # Only compare different classes
            if det1['label'] != det2['label']:
                # Calculate IoU
                box1 = det1['bbox']
                box2 = det2['bbox']
                xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
                xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > iou_threshold:
                    # Keep only the higher confidence detection
                    if det1['confidence'] < det2['confidence']:
                        keep = False
                        break
        if keep:
            filtered.append(det1)
            used.add(i)
    return filtered

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generate_checklist', methods=['POST'])
def generate_checklist():
    if 'reference' not in request.files:
        return jsonify({'error': 'Missing reference image'}), 400
    reference = request.files['reference']
    if not (reference and allowed_file(reference.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f'ref_{timestamp}.jpg')
    reference.save(ref_path)
    detections = yolo_detect(ref_path)
    # Filter out 'person' detections
    detections = [d for d in detections if d['label'] != 'person']
    detections = remove_duplicate_detections(detections)
    detections = suppress_overlapping_classes(detections)
    vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f'ref_vis_{timestamp}.jpg')
    draw_boxes(ref_path, detections, vis_path)
    checklist = []
    seen = set()
    for d in detections:
        label = d['label']
        if label not in seen:
            seen.add(label)
            instruction = CLEANING_INSTRUCTIONS.get(label, f"{label} Detected. If unsure, refer to property guidelines.")
            checklist.append({
                'item': label,
                'instruction': instruction
            })
    return jsonify({
        'checklist': checklist,
        'detections': detections,
        'visualization': os.path.basename(vis_path)
    })

@app.route('/compare', methods=['POST'])
def compare():
    if 'reference' not in request.files or 'current' not in request.files:
        return jsonify({'error': 'Missing image files'}), 400
    reference = request.files['reference']
    current = request.files['current']
    if not (reference and current and allowed_file(reference.filename) and allowed_file(current.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f'ref_{timestamp}.jpg')
    curr_path = os.path.join(app.config['UPLOAD_FOLDER'], f'curr_{timestamp}.jpg')
    reference.save(ref_path)
    current.save(curr_path)
    ref_detections = yolo_detect(ref_path)
    curr_detections = yolo_detect(curr_path)
    # Filter out 'person' detections
    ref_detections = [d for d in ref_detections if d['label'] != 'person']
    curr_detections = [d for d in curr_detections if d['label'] != 'person']
    ref_detections = remove_duplicate_detections(ref_detections)
    curr_detections = remove_duplicate_detections(curr_detections)
    ref_detections = suppress_overlapping_classes(ref_detections)
    curr_detections = suppress_overlapping_classes(curr_detections)
    ref_vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f'ref_vis_{timestamp}.jpg')
    curr_vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f'curr_vis_{timestamp}.jpg')
    draw_boxes(ref_path, ref_detections, ref_vis_path)
    draw_boxes(curr_path, curr_detections, curr_vis_path)
    ref_labels = set([d['label'] for d in ref_detections])
    curr_labels = set([d['label'] for d in curr_detections])
    missing = ref_labels - curr_labels
    new = curr_labels - ref_labels
    issues = []
    overlay_img = cv2.imread(curr_path)
    COLOR_MISSING = (0, 0, 255)
    COLOR_DIRTY = (0, 165, 255)
    COLOR_OK = (0, 200, 0)
    overlay_legend = [
        (COLOR_MISSING, 'Missing'),
        (COLOR_DIRTY, 'Dirty/Damaged'),
        (COLOR_OK, 'OK')
    ]
    # Track which after-image detections have been matched
    matched_after_indices = set()

    for r_idx, rbox in enumerate(ref_detections):
        label = rbox['label']
        best_iou = 0
        best_c_idx = -1
        for c_idx, cbox in enumerate(curr_detections):
            if cbox['label'] != label or c_idx in matched_after_indices:
                continue
            # Calculate IoU
            rx1, ry1, rx2, ry2 = rbox['bbox']
            cx1, cy1, cx2, cy2 = cbox['bbox']
            xi1, yi1 = max(rx1, cx1), max(ry1, cy1)
            xi2, yi2 = min(rx2, cx2), min(ry2, cy2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            r_area = (rx2 - rx1) * (ry2 - ry1)
            c_area = (cx2 - cx1) * (cy2 - cy1)
            union_area = r_area + c_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_c_idx = c_idx
        if best_iou > 0.2:
            matched_after_indices.add(best_c_idx)
            # Area ratio logic for dirty/OK as before
            cbox = curr_detections[best_c_idx]
            r_area = (rbox['bbox'][2] - rbox['bbox'][0]) * (rbox['bbox'][3] - rbox['bbox'][1])
            c_area = (cbox['bbox'][2] - cbox['bbox'][0]) * (cbox['bbox'][3] - cbox['bbox'][1])
            area_ratio = c_area / r_area if r_area > 0 else 0
            if area_ratio < 0.5:
                issues.append(f"{label.capitalize()} appears smaller or partially missing (possible damage or removal).")
                cx1, cy1, cx2, cy2 = cbox['bbox']
                cv2.rectangle(overlay_img, (cx1, cy1), (cx2, cy2), COLOR_DIRTY, 3)
                cv2.putText(overlay_img, f"{label} (dirty?)", (cx1, max(cy1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DIRTY, 2)
            else:
                cx1, cy1, cx2, cy2 = cbox['bbox']
                cv2.rectangle(overlay_img, (cx1, cy1), (cx2, cy2), COLOR_OK, 2)
                cv2.putText(overlay_img, f"{label}", (cx1, max(cy1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_OK, 2)
        else:
            issues.append(f"{label.capitalize()} is missing or moved.")
            rx1, ry1, rx2, ry2 = rbox['bbox']
            cv2.rectangle(overlay_img, (rx1, ry1), (rx2, ry2), COLOR_MISSING, 3)
            cv2.putText(overlay_img, f"{label} (missing)", (rx1, max(ry1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MISSING, 2)

    # Now, for each after-image detection not matched, mark as new
    for c_idx, cbox in enumerate(curr_detections):
        if c_idx not in matched_after_indices:
            label = cbox['label']
            issues.append(f"New {label} detected.")
            cx1, cy1, cx2, cy2 = cbox['bbox']
            cv2.rectangle(overlay_img, (cx1, cy1), (cx2, cy2), COLOR_MISSING, 3)
            cv2.putText(overlay_img, f"{label} (new)", (cx1, max(cy1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MISSING, 2)
    if not issues:
        issues.append("Great job! No significant issues detected.")
    legend_x, legend_y = 20, 40
    for color, text in overlay_legend:
        cv2.rectangle(overlay_img, (legend_x, legend_y-20), (legend_x+30, legend_y+5), color, -1)
        cv2.putText(overlay_img, text, (legend_x+40, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        legend_y += 35
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f'overlay_{timestamp}.jpg')
    cv2.imwrite(overlay_path, overlay_img)
    return jsonify({
        'issues': issues,
        'ref_visualization': os.path.basename(ref_vis_path),
        'curr_visualization': os.path.basename(curr_vis_path),
        'overlay_visualization': os.path.basename(overlay_path)
    })

if __name__ == '__main__':
    app.run(debug=True) 