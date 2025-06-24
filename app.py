from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
from ai import yolo_detect, draw_boxes, remove_duplicate_detections, compare, suppress_overlapping_classes, allowed_file, CLEANING_INSTRUCTIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
            checklist.append({'item': label, 'instruction': instruction})
    return jsonify({
        'checklist': checklist,
        'detections': detections,
        'visualization': os.path.basename(vis_path)
    })

@app.route('/compare', methods=['POST'])
def compare_route():
    return compare()

if __name__ == '__main__':
    app.run(debug=True)
