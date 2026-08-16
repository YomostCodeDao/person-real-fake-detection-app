# SRC USAGE INSTRUCTIONS

**DOWNLOAD BOTH MODELS FIRST BEFORE RUNNING CODE, THEN PROCEED WITH THE NEXT STEPS** *Download link:* [Google Drive](https://drive.google.com/drive/folders/1PKR0FWo0GmaohdHH9aLMOK_Lg5Z7Em0Q?usp=sharing/)

# 1. Open terminal (Switch to Git Bash) and check the current directory

- If in `/src`, no action needed
- If in `/project`, run `cd src`

# 2. Run Commands
**FOR FIRST TIME SETUP, RUN STEP 2.1**
- 2.1. `python -m venv venv` (Create virtual environment - creates a `venv` folder in `src`)
- 2.2. `source venv/Scripts/activate` (Activate virtual environment - `.venv` will appear in the prompt)
- 2.3. `pip install -r requirements.txt` (Install required packages - if an error occurs, check the path to `requirements.txt`)
- 2.4. `python app.py` (Run this inside the terminal with virtual environment activated)

**If started successfully, the Flask log will appear. Open your browser at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

========

# Key Features

## Image Upload

## Select an image from your device → the system will:

### Run YOLO for object detection.

### Run custom EfficientNet-B4 for REAL/FAKE classification.

### Display original image, image with bounding boxes, YOLO results table, and the REAL/FAKE verdict.

## Realtime Webcam

## Open webcam → the system will:

### Detect only the `person` class.

### Overlay REAL/FAKE result + person label directly onto the video feed.
### Display 2 lines of information below the video:
#### REAL/FAKE verdict
#### Detected entity (`person` / no person detected)

========
# PROJECT STRUCTURE
src/
│
├── app.py                          # Main Flask server
├── templates/
│ ├── index.html                    # Image upload interface
│ └── webcam.html                   # Realtime webcam interface
├── static/
│ └── uploads/                      # Uploaded images and YOLO results
├── requirements.txt                # Required Python packages
├── best_face_fake_detector14b211.pth # Custom EfficientNet-B4 checkpoint
└── yolo11n.pt                      # YOLO model

