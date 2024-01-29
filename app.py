import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__, static_folder='static')
recognized_text = ""
recognition_process = None

@app.route("/")
def index():
    return render_template("st_page.html")

@app.route("/start-recognition", methods=["POST"])
def start_recognition():
    global recognition_process
    recognition_process = subprocess.Popen(["google-chrome", "--new-window", "http://localhost:5000"], shell=True)
    return jsonify({"status": "success"})

@app.route("/stop-recognition", methods=["POST"])
def stop_recognition():
    global recognition_process
    recognition_process.terminate()
    return jsonify({"status": "success"})

@app.route("/process-text", methods=["POST"])
def process_text():
    global recognized_text
    recognized_text = request.json["text"]
    return jsonify({"status": "success"})

@app.route("/upload-video", methods=["POST"])
def upload_video():
    global recognized_text
    if recognized_text:
        print(recognized_text)
        video_filename = f"combined_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join("static", video_filename)
        subprocess.run(["ffmpeg", "-i", "static\intent.mp4", "-i", "static\hearing.mp4", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", video_path])
        return jsonify({"status": "success", "video_url": f"/{video_filename}"})
    else:
        return jsonify({"status": "error", "message": "No recognized text provided"})

@app.route("/video/<path:filename>")
def send_video(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)