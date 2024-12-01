import cv2
import mss
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

def gen_frames():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Change index if you have multiple monitors
        while True:
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Optionally resize the image to reduce bandwidth
            # img = cv2.resize(img, (800, 600))

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            # Yield the output frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Simple page to display the video feed
    return '''
    <html>
    <head>
        <title>Real-Time Screen Streaming</title>
    </head>
    <body>
        <h1>Real-Time Screen Streaming</h1>
        <img src="/video_feed" width="100%">
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the app on the local network
    app.run(host='0.0.0.0', port=5001, threaded=True)