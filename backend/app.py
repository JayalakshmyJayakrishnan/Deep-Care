import os
import sys
import cv2
import numpy as np
import flask
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add project root to path to ensure imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.core.pipeline import DeepCarePipeline

# Production: Serve React Build
DIST_FOLDER = os.path.join(os.getcwd(), 'frontend/dist')
if not os.path.exists(DIST_FOLDER):
    DIST_FOLDER = os.path.join(os.getcwd(), '..', 'frontend', 'dist')

app = Flask(__name__, static_folder=os.path.join(DIST_FOLDER, 'assets'), template_folder=DIST_FOLDER)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Pipeline
pipeline = DeepCarePipeline()

@app.route('/')
def serve_react_app():
    return flask.render_template('index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    if path.startswith('uploads/'):
        return send_from_directory('.', path)
    return flask.send_from_directory(DIST_FOLDER, path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return flask.send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Deep Care AI Backend"})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Determine if video or image
    # For this simplified demo, we treat everything as a "video" source 
    # even if it's a single image (the extract_frames handles this if adapted, 
    # but strictly it uses VideoCapture which usually handles images too).
    # But for robustness, let's verify extension.
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        # It's an image, create a fake 1-frame video logic or just process it directly
        # For consistency with the "Video Sequence" requirement, we could just process it as a single frame.
        pass
    
    # Run the Deep Care Pipeline
    try:
        results = pipeline.process_video(filepath, UPLOAD_FOLDER)
        
        # Aggregate stats
        total_frames = len(results)
        skipped = len([r for r in results if r['status'] == 'skipped'])
        abnormal = len([r for r in results if r.get('prediction', {}).get('pathology') != 'Normal' and r['status'] == 'processed'])
        
        return jsonify({
            "message": "Processing complete",
            "filename": file.filename,
            "summary": {
                "total_frames_processed": total_frames,
                "frames_skipped": skipped,
                "abnormalities_detected": abnormal
            },
            "detailed_results": results
        })
    except Exception as e:
        print(f"Error processing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/demo', methods=['POST'])
def run_demo():
    from backend.utils.dataset_helper import DatasetHelper
    import shutil
    
    helper = DatasetHelper('galar')
    sample_path = helper.get_random_sample()
    
    if not sample_path:
        return jsonify({"error": "No sample data found. Please run 'backend/scripts/download_galar.py' first."}), 404
        
    # Copy to uploads to simulate a user upload
    filename = os.path.basename(sample_path)
    target_path = os.path.join(UPLOAD_FOLDER, "DEMO_" + filename)
    shutil.copy(sample_path, target_path)
    
    try:
        results = pipeline.process_video(target_path, UPLOAD_FOLDER)
        
        # Aggregate stats
        total_frames = len(results)
        skipped = len([r for r in results if r['status'] == 'skipped'])
        abnormal = len([r for r in results if r.get('prediction', {}).get('pathology') != 'Normal' and r['status'] == 'processed'])
        
        return jsonify({
            "message": "Demo processed successfully",
            "filename": filename,
            "is_demo": True,
            "summary": {
                "total_frames_processed": total_frames,
                "frames_skipped": skipped,
                "abnormalities_detected": abnormal
            },
            "detailed_results": results
        })
    except Exception as e:
        print(f"Error processing demo: {e}")
        return jsonify({"error": str(e)}), 500

import uuid
import base64
import datetime

STORE = {
    "users": [
        {
            "hospitalId": "HOSP-001",
            "hospName": "General Hospital",
            "staffId": "STAFF-001",
            "firstName": "Demo",
            "lastName": "Doctor",
            "role": "Gastroenterologist",
            "passwordHash": base64.b64encode("password123".encode('utf-8')).decode('utf-8')
        }
    ],
    "patients": [],
    "sessions": {},
    "_nextPatId": 1
}

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    
    for u in STORE["users"]:
        if u.get("staffId") == data.get("staffId") and u.get("hospitalId") == data.get("hospitalId"):
            return jsonify({"error": "Staff ID already registered for this hospital"}), 400
            
    password_bytes = data.get("password", "").encode('utf-8')
    password_hash = base64.b64encode(password_bytes).decode('utf-8')
    
    user = data.copy()
    user["passwordHash"] = password_hash
    STORE["users"].append(user)
    
    token = 'live_' + str(uuid.uuid4())
    STORE["sessions"][token] = user
    return jsonify({"token": token, "user": user})

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    password_bytes = data.get("password", "").encode('utf-8')
    password_hash = base64.b64encode(password_bytes).decode('utf-8')
    
    for u in STORE["users"]:
        if (u.get("hospitalId") == data.get("hospitalId") and 
            u.get("staffId") == data.get("staffId") and 
            u.get("passwordHash") == password_hash):
            token = 'live_' + str(uuid.uuid4())
            STORE["sessions"][token] = u
            return jsonify({"token": token, "user": u})
            
    return jsonify({"error": "Invalid credentials. Please check your Hospital ID, Staff ID and password."}), 401

@app.route('/api/patients', methods=['GET', 'POST'])
def patients():
    if request.method == 'POST':
        data = request.json
        padded = str(STORE["_nextPatId"]).zfill(4)
        pat_id = data.get("patientId") or f"PAT-{datetime.date.today().year}-{padded}"
        STORE["_nextPatId"] += 1
        
        dob_str = data.get("dob", "")
        age = 0
        if dob_str:
            try:
                dob = datetime.datetime.strptime(dob_str, "%Y-%m-%d").date()
                today = datetime.date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except:
                pass
                
        patient = {
            "id": pat_id, 
            "name": f"{data.get('firstName', '')} {data.get('lastName', '')}".strip(),
            "age": age, 
            "gender": data.get("gender") or "—", 
            "doctor": data.get("doctor"),
            "date": data.get("studyDate"), 
            "status": "pending", 
            "insId": data.get("insId")
        }
        STORE["patients"].append(patient)
        return jsonify({"patient": patient})
    
    return jsonify({"patients": STORE["patients"]})

@app.route('/api/studies/analyze', methods=['POST'])
def analyze_study():
    return jsonify({"studyId": f"STUDY-{int(datetime.datetime.now().timestamp() * 1000)}", "message": "Pipeline started"})

@app.route('/api/reference/categories', methods=['GET'])
def get_reference_categories():
    # Return pathology classes defined in MTLEngine
    from backend.models.mtl_engine import MTLEngine
    engine = MTLEngine()
    response = jsonify({"categories": engine.lesion_classes})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/api/reference/sample', methods=['GET'])
def get_reference_sample():
    pathology = request.args.get('pathology', 'Normal')
    mount_path = os.environ.get('GALAR_MOUNT_PATH')
    
    if not mount_path or not os.path.exists(mount_path):
        # Fallback to demo mode if hospital NAS isn't mounted
        from backend.utils.dataset_helper import DatasetHelper
        helper = DatasetHelper('galar')
        sample_path = helper.get_random_sample()
        if not sample_path:
            resp = jsonify({"error": "No reference data found. Is GALAR_MOUNT_PATH configured?"})
            resp.headers.add("Access-Control-Allow-Origin", "*")
            return resp, 404
            
        import shutil
        filename = os.path.basename(sample_path)
        target_path = os.path.join(UPLOAD_FOLDER, "REF_" + filename)
        if not os.path.exists(target_path):
            shutil.copy(sample_path, target_path)
            
        resp = jsonify({"sampleUrl": f"/uploads/REF_{filename}", "pathology": "Unknown (Unmounted Dataset)", "source": "Limited Fallback Dataset"})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp
        
    resp = jsonify({"error": f"Pathology '{pathology}' queried, but metadata indexing is still building on {mount_path}."})
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp, 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
