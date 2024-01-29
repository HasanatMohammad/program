from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# OpenCV VideoCapture
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/start_camera')
def start_camera():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    camera.release()
    cv2.destroyAllWindows()
    return 'Camera stopped'

if __name__ == "__main__":
    app.run(debug=True)