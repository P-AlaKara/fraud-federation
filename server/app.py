import os
from flask import Flask, request, send_file, jsonify
from aggregator import perform_aggregation

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
MODELS_FOLDER = './models'
GLOBAL_MODEL_NAME = 'global_model.ckpt'
# Trigger aggregation after receiving updates from this many banks
AGGREGATION_THRESHOLD = 2 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# --- Endpoints ---

@app.route('/api/model/latest', methods=['GET'])
def get_latest_model():
    """
    Spoke -> Hub: Bank requests the latest global model.
    """
    global_model_path = os.path.join(MODELS_FOLDER, GLOBAL_MODEL_NAME)
    
    if os.path.exists(global_model_path):
        return send_file(global_model_path, as_attachment=True)
    else:
        return jsonify({"error": "Global model not initialized yet."}), 404

@app.route('/api/model/upload', methods=['POST'])
def upload_update():
    """
    Spoke -> Hub: Bank uploads their local training weights.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    bank_id = request.headers.get('Bank-ID', 'unknown_bank')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save file as bank_id.ckpt
        filename = f"{bank_id}.ckpt"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        
        # Check if we should trigger aggregation
        current_uploads = len([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.ckpt')])
        
        msg = f"Upload successful. Pending: {current_uploads}/{AGGREGATION_THRESHOLD}"
        
        if current_uploads >= AGGREGATION_THRESHOLD:
            print("Threshold met. Triggering Aggregator...")
            perform_aggregation()
            msg = "Upload successful. Aggregation Triggered."
            
        return jsonify({"message": msg, "status": "accepted"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)