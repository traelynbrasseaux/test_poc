# Keepers Test

A Flask web application for automated property cleaning checklists and before/after photo comparison using YOLOv8 object detection.

## Features
- **Checklist Generation:** Upload a reference photo to generate a cleaning checklist based on detected objects (kitchenware, furniture, appliances, etc.).
- **Before/After Comparison:** Upload reference and current photos to detect missing, new, or dirty/damaged items.
- **Visual Feedback:** Annotated images with bounding boxes and status overlays.
- **Custom Cleaning Instructions:** Context-aware instructions for each detected object.

## How It Works
- Uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models (COCO dataset) for object detection.
- Provides cleaning instructions for each detected item.
- Compares before/after images to highlight changes and issues.

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/traelynbrasseaux/test_poc.git
   cd test_poc
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Download YOLOv8 model files:**
   - The `.pt` model files (`yolov8x.pt`, `yolov8l.pt`, `yolov8m.pt`, `yolov8s.pt`, `yolov8n.pt`) are tracked with [Git LFS](https://git-lfs.github.com/).
   - If you clone with Git LFS installed, these files will be downloaded automatically. If not, install Git LFS and run:
     ```sh
     git lfs pull
     ```

## Usage
1. **Start the Flask app:**
   ```sh
   python app.py
   ```
2. **Open your browser:**
   - Go to `http://127.0.0.1:5000/`
3. **Generate a checklist:**
   - Upload a reference photo to get a list of detected items and cleaning instructions.
4. **Compare before/after photos:**
   - Upload both reference and current photos to see what's missing, new, or possibly dirty/damaged.

## File Structure
- `app.py` — Main Flask application
- `requirements.txt` — Python dependencies
- `templates/index.html` — Web UI
- `uploads/` — Uploaded and processed images
- `yolov8*.pt` — YOLOv8 model weights (large files, handled by Git LFS)

## Notes
- **Large Files:**
  - Model files are large and managed with Git LFS. If you don't see them after cloning, install Git LFS and run `git lfs pull`.
- **Customization:**
  - You can add or modify cleaning instructions in the `CLEANING_INSTRUCTIONS` dictionary in `app.py`.

## License
This project is for demonstration and internal use. Please contact the author for licensing details. 