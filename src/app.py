# src/app.py
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request, redirect, Response, url_for, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
import timm
from ultralytics import YOLO

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# AntiSpoof Model (EfficientNet-B4 backbone)
# ============================
class AntiSpoofModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b2", pretrained=False)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)   # output 1 logit
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# Load YOLO + AntiSpoof model
model_yolo = YOLO('yolo11n.pt')

model_antispoof = AntiSpoofModel().to(device)
ckpt_path = 'best_face_fake_detector14b211.pth'
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    model_antispoof.load_state_dict(state, strict=False)  # cho phép thiếu/thừa keys
    print(f"=== Đã load model EfficientNet-B4: {ckpt_= path}")
    print("=== Đã load model YOLOv11")
else:
    print("⚠️ Không tìm thấy checkpoint, model chưa được load!")


model_antispoof.eval()

# Transform cho inference
infer_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ============================
# Hàm phụ trợ
# ============================
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_yolo(image_path: str, save_plotted: bool = True):
    img = Image.open(image_path)
    results = model_yolo(img)

    plotted_filename = None
    if save_plotted and len(results) > 0:
        plotted = results[0].plot()
        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        plotted_filename = f"{name}_yolo{ext}"
        plotted_path = os.path.join(app.config['UPLOAD_FOLDER'], plotted_filename)
        Image.fromarray(plotted).save(plotted_path)

    try:
        df = results[0].to_df()
        if hasattr(df, "to_dicts"):
            records = df.to_dicts()
        else:
            records = df.to_dict(orient='records')
        if len(records) == 0:
            return {"message": "Không phát hiện đối tượng nào", "records": [], "plotted_filename": plotted_filename}
        return {"message": None, "records": records, "plotted_filename": plotted_filename}
    except Exception as e:
        return {"message": f"Lỗi YOLO: {e}", "records": [], "plotted_filename": plotted_filename}

def predict_real_fake(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img_tensor = infer_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_antispoof(img_tensor)
        prob = torch.sigmoid(output).item()
        verdict = "REAL" if prob >= 0.5 else "FAKE"

    return verdict, prob

# ============================
# Webcam xử lý realtime
# ============================
camera = cv2.VideoCapture(0)

# Biến toàn cục để lưu kết quả realtime
latest_verdict = "N/A"
latest_prob = 0.0
latest_entity = "N/A"

def gen_frames():
    global latest_verdict, latest_prob, latest_entity
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # YOLO detect
            results = model_yolo(frame)
            entities = []
            if len(results) > 0:
                df = results[0].to_df()
                if hasattr(df, "to_dicts"):
                    records = df.to_dicts()
                else:
                    records = df.to_dict(orient='records')
                # chỉ lấy nhãn person
                for r in records:
                    if r['name'] == 'person':
                        entities.append('person')
            latest_entity = "person" if entities else "Khong phat hien nguoi"

            # EfficientNet-B4 custom
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = infer_tfms(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model_antispoof(img_tensor)
                prob = torch.sigmoid(output).item()
                verdict = "REAL" if prob >= 0.5 else "FAKE"

            latest_verdict = verdict
            latest_prob = prob

            # Overlay lên frame
            cv2.putText(frame, f"{verdict} ({prob:.2f})", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, f"Entity: {latest_entity}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ============================
# Flask routes
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    yolo_output = predict_yolo(file_path, save_plotted=True)
    verdict, prob = predict_real_fake(file_path)

    return render_template(
        'index.html',
        filename=filename,
        yolo_message=yolo_output.get("message"),
        yolo_results=yolo_output.get("records"),
        yolo_plotted=yolo_output.get("plotted_filename"),
        verdict=verdict,
        prob=prob
    )

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# API để lấy kết quả realtime (cho webcam.html hiển thị thêm)
@app.route('/webcam_status')
def webcam_status():
    return jsonify({
        "verdict": latest_verdict,
        "prob": latest_prob,
        "entity": latest_entity
    })

if __name__ == '__main__':
    app.run(debug=True)
