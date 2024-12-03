import cv2
import mss
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

def gen_frames():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Change index if you have multiple monitors
        while True:
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Apply object detection on the frame
            results = model.predict(img, conf=0.8, verbose = False)

            # Get the annotated frame from the results
            annotated_frame = results[0].plot()

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
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