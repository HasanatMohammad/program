from flask import Flask, render_template, request, jsonify
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('st_page.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    text = request.form['text']

    # Save the text to a file (e.g., text.txt)
    with open('text.txt', 'w') as file:
        file.write(text)

    # Assume you have another video file named output2.mp4
    video_clip1 = VideoFileClip("static\intent.mp4")
    video_clip2 = VideoFileClip("static\hearing.mp4")

    # Combine the video clips
    combined_clip = concatenate_videoclips([video_clip1, video_clip2])

    # Write the combined clip to a new file in the static folder
    combined_clip_path = os.path.join('static', 'combined.mp4')
    combined_clip.write_videofile(combined_clip_path, codec="libx264", audio_codec="aac")

    # Send a message to the HTML page along with the combined MP4 path
    return jsonify({'message': 'Combined video ready', 'combined_mp4_path': combined_clip_path})

if __name__ == '__main__':
    app.run(debug=True)