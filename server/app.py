from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, TFAutoModelForSeq2SeqLM, Trainer, TrainingArguments
import uuid
import os
import requests
os.environ["PATH"] = os.pathsep.join([
    p for p in os.environ["PATH"].split(os.pathsep)
    if "nvidia" not in p.lower() and "cudnn" not in p.lower()
])
from server.config_generate import get_Config_config
from flask_cors import CORS
from pydub import AudioSegment
from faster_whisper import WhisperModel, BatchedInferencePipeline
from speech_to_text_fw import speech_to_text

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    
executor = ThreadPoolExecutor(max_workers=5)
tasks = {}
lock = threading.Lock()
gpu_lock = threading.Lock()

user_input =  "I want a animated look with digital time, a subtle date box at six o'clock, health Widgets around all the edges and elegant cyan circle Widget color with moonphase. Mouth out save report want box recognize strong. Work individual he then candidate easy here specific."
"I want a timeless look with Roman numerals, a subtle date box at six o'clock, and elegant gold hands on a deep leather-brown background. Mouth out save report want box recognize strong. Work individual he then candidate easy here specific."

# @app.route('/', methods=["GET"])
# def index():
#     return "Flask server is running"

# @app.route('/create/config/from/audio', methods=["POST"])
# def create_Config_from_audio():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file part'}), 400
#     file = request.files.get('audio')
#     if file:
#         model = whisper.load_model("medium") 
#         save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(save_path)
#         # audio = AudioSegment.from_file(save_path, format="3gp")
#         # converted_path = f"{UPLOAD_FOLDER}/converted.wav"
#         # audio.export(converted_path, format="wav")
#         result = model.transcribe(save_path)
#         print(result["text"])
#         host = request.host_url.rstrip("/")
#         generated_output, generated_image_url = get_Config_config(result["text"], host)
#         print(generated_image_url)
#         return generated_output
#     return jsonify({"error": "No file received"}), 400

@app.route('/create/config/from/audio', methods=["POST"])
def create_Config_from_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    file = request.files.get('audio')
    if file:
        text = speech_to_text(file)
        print(text)
        host = request.host_url.rstrip("/")
        generated_output, generated_image_url = get_Config_config(text, host)
        print(generated_output)
        return jsonify({
            "text": text,
            "output": generated_output
        })
    return jsonify({"error": "No file received"}), 400

def update_task_status(task_id, status, progress, result=None, error=None):
    with lock:
        tasks[task_id] = {
            "status": status,
            "progress": progress,
            "result": result,
            "error": error
        }
imageURL = ""
def multi_task(user_input, host, task_id):
    try:
        update_task_status(task_id, "Queued", progress=5)
        def model_progress(p):
            update_task_status(task_id, "Running", p)
        with gpu_lock:
            print(f"[GPU] processing task {task_id} for: {user_input}")
            generated_output, generated_image_url = get_Config_config(user_input, host)
            print(generated_output)
            print(generated_image_url)
            imageURL = generated_image_url
            result = {
                "status": "Success",
                "output": generated_output
            }
            update_task_status(task_id, "Completed",progress=100, result=generated_output)
    except Exception as e:
        update_task_status(task_id, "Failed",progress=100, error=str(e))

@app.route('/create/config/from/text', methods=["POST"])
def create_Config_from_text():
    data = request.get_json(force=True)
    user_input = data.get("inputText", "").strip()
    if not user_input:
        return jsonify({"error": "No input text provided"}), 400
    host = request.host_url.rstrip("/")
    task_id = str(uuid.uuid4())
    update_task_status(task_id, "Queued", progress=0)
    executor.submit(multi_task, user_input, host, task_id)
    # print(generated_image_url)
    # return generated_output
    return jsonify({
        "task_id": task_id,
        "status": "Queued"
    })
        
@app.route('/status/<task_id>', methods=["GET"])
def status(task_id):
    with lock:
        task_info = tasks.get(task_id)
        if not task_info:
            return jsonify({
            "error": "Invalid task ID"
        }), 404
        return jsonify(task_info)

@app.route('/result/<task_id>', methods=["GET"])
def result(task_id):
    task_info = tasks.get(task_id)
    if not task_info:
        return jsonify({
        "error": "Invalid task ID"
    }), 404
    if task_info["status"] != "Completed":
        return jsonify({
            "status": task_info["status"]
        })
    return jsonify(task_info)

BASE_URL = "http://192.168.0.68:5000/"

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["input_text"]
        if not user_input:
            return jsonify({"error": "No input text provided"}), 400
        host = request.host_url.rstrip("/")
        task_id = str(uuid.uuid4())
        update_task_status(task_id, "Queued", progress=0)
        executor.submit(multi_task, user_input, host, task_id)
        # print(generated_image_url)
        # return generated_output
        while True:
            status_result = requests.get(f"{BASE_URL}/status/{task_id}")
            if status_result.status_code == 200:
                status_info = status_result.json()
                print(f"Task {task_id} status: {status_info}")
                if status_info["status"] in ["Completed", "Failed"]:
                    return render_template('output.html', user_input=user_input, generated_output=status_info["result"], generated_image=imageURL)
                    break
            else:
                print(f"Error while check task status for task_id {task_id}")
            time.sleep(3)
        
    return render_template('index.html') 


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)