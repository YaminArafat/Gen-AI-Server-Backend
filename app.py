# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os
import whisper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app) 

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return "Flask server is running."

def process_input():
    output = json.load(open('.\datasets\demo.json'))
    return output

@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.get_json(force=True)
    input_text = data.get("inputText", "").strip()
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400
    json_text = process_input()

    return json_text

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    file = request.files.get('audio')
    if file:
        model = whisper.load_model("medium") 
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        result = model.transcribe(save_path)
        print(result["text"])
        return jsonify({"message": "File uploaded", "path": save_path, "text": result["text"]})
    return jsonify({"error": "No file received"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 
