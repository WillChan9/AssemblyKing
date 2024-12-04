import cv2
import mss
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import os
import openai # Import the OpenAI library
import time
import base64
import threading

app = Flask(__name__)
model = YOLO('best.pt')

# Global variables
instructions_text = ''

# Helper function to encode images to base64
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

def generate_instructions(image1, image2):
    Asembly_instruction = '''The user is doing a assembly task involved 6 parts, that would be use for assembling a mechanical dice. 
The Mechanical Dice is a pocket watch-style 3D-printed device with a spinning internal die wheel (faces 1-6) activated by a top button. 
Pressing the button spins the wheel, and releasing it stops it to display a random number. 
Key parts include the circular bottom and top case, a die wheel, a spring, a plunger button, a push-down trigger. 

To assemble: mount the spring on the extruder on the bottom plate, plug the trigger on the axle, then push down the trigger to make room for die wheel, seal the case with the top plate, and insert the button on the trigger. 
    '''

    openai.api_key = os.getenv('OPENAI_API_KEY')
    API_RESPONSE1 = openai.chat.completions.create(
        model="gpt-4o",  # Ensure this model is capable of processing images
        messages=[
            {
                "role": "system",
                "content": Asembly_instruction
            },
            {
                "role": "user",
                "content": [ # Changed this line to include content as a list
                    {
                        "type": "text",
                        "text": '''base on user's current view of assembly and the object in the scene, generate a brief instructions for the user.'''
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image1)}" # Changed to jpeg
                        }
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image2)}" # Changed to jpeg
                        }
                    }
                ]
            }
        ]
    )

    instructions = API_RESPONSE1.choices[0].message.content
    return instructions

def sample_frames_and_generate_instructions():
    global instructions_text

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Change index if you have multiple monitors

        while True:
            # Sample frames every 3 seconds
            images = []
            for _ in range(2):
                # Capture the screen
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                images.append(img)
                time.sleep(5)  # Wait for 3 seconds

            # Now we have 2 images collected over 10 seconds
            # Generate instructions
            instructions = generate_instructions(images[0], images[1])

            # Update the global instructions_text variable
            instructions_text = instructions

def gen_frames():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Change index if you have multiple monitors
        while True:
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Apply object detection on the frame
            results = model.predict(img, verbose=False)

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
    # Simple page to display the video feed and instructions
    return '''
    <html>
    <head>
        <title>Real-Time Screen Streaming</title>
        <script>
            function fetchInstructions() {
                fetch('/instructions')
                    .then(response => response.text())
                    .then(data => {
                        document.getElementById('instructions').innerText = data;
                    });
            }
            setInterval(fetchInstructions, 9000);  // Fetch instructions every 9 seconds
            window.onload = fetchInstructions;  // Fetch instructions on page load
        </script>
    </head>
    <body>
        <h1>Real-Time Screen Streaming</h1>
        <img src="/video_feed" width="100%">
        <h2>Instructions:</h2>
        <div id="instructions"></div>
    </body>
    </html>
    '''

@app.route('/instructions')
def get_instructions():
    global instructions_text
    return instructions_text

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the background thread
    instruction_thread = threading.Thread(target=sample_frames_and_generate_instructions)
    instruction_thread.daemon = True
    instruction_thread.start()

    # Run the app on the local network
    app.run(host='0.0.0.0', port=5001, threaded=True)