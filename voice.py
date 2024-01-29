# voice.py
import os
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from moviepy.editor import TextClip, VideoFileClip, concatenate_videoclips

app = Flask(__name__)

recognizer = sr.Recognizer()
microphone = sr.Microphone()

words = ['clean', 'hearing', 'intent', 'learn', 'like_love', 'meet', 'name', 'no', 'sign', 'slow', 'slowly',
         'student', 'teacher', 'what', 'where', 'who', 'why', 'yes', 'your']


@app.route('/')
def index():
    return render_template('v_page.html')

from base64 import b64encode
@app.route('/start_microphone', methods=['POST'])
def start_microphone():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source, timeout=10)

    audio_base64 = b64encode(audio_data.get_wav_data()).decode('utf-8')

    return jsonify({'status': 'success', 'audio_base64': audio_base64})

@app.route('/start_microphone', methods=['POST'])
def start_microphone():
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source, timeout=10)

        audio_base64 = b64encode(audio_data.get_wav_data()).decode('utf-8')

        return jsonify({'status': 'success', 'audio_base64': audio_base64})
    except Exception as e:
        print(f"Error in /start_microphone: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Error processing audio'})


def create_video(sentences):
    videos = []

    sentences = sentences.replace("?", "")
    if "," not in sentences:
        bm = 0
    else:
        bm = sentences.count(",")

    for i in range(bm + 1):
        if sentences.lower() == "are you student":
            videos.append(os.path.join(r"data\\", "are you student" + ".mp4"))

        elif sentences.lower() in ["what's your name", "what is your name"]:
            videos.append(os.path.join(r"data\\", "what your name" + ".mp4"))

        elif sentences.lower() in ["don't", "donot", "didnot", "didn't"] and sentences.lower() in [
            "grasp", "understand", "comprehend", "gotit"]:
            videos.append(os.path.join(r"data\\", "don't understand" + ".mp4"))

        elif sentences.lower() == "are you deaf":
            videos.append(os.path.join(r"data\\", "deaf you" + ".mp4"))

        elif sentences.lower() in ["your", "yours", "belongtoyou", "toyou"]:
            videos.append(os.path.join(r"data\\", "your" + ".mp4"))

        else:
            sentences = sentences + " "
            for i in range(sentences.count(" ") + 1):
                bm = sentences.index(" ")
                word = sentences[:bm]
                if word in words:
                    videos.append(os.path.join(r"data\\", word + ".mp4"))
                sentences = sentences[bm + 1:]

    if len(videos) == 1:
        path = videos[0]
    else:
        final_video = concatenate_videoclips([VideoFileClip(video_path) for video_path in videos], method="compose")
        final_video.write_videofile(r"video\specific.mp4")
        path = r"video\specific.mp4"

    return path


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)