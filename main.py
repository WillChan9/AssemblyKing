import cv2
import mss
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import os
import openai  # Import the OpenAI library
import time
import base64
import threading

app = Flask(__name__)
model = YOLO('best.pt')

# Global variables
instructions_text = ''
current_objects = []

# Helper function to encode images to base64
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

def generate_instructions(image1, image2, image3, detected_objects):
    assembly_instruction = '''The user is performing an assembly task involving 6 parts to assemble a mechanical dice. 
The Mechanical Dice is a pocket watch-style 3D-printed device with a spinning internal die wheel (faces 1-6) activated by a top button. 
Pressing the button spins the wheel, and releasing it stops it to display a random number. 
Key parts include the circular bottom and top case, a die wheel, a spring, a plunger button, a push-down trigger. 

To assemble: mount the spring on the extruder on the bottom plate, plug the trigger on the axle, then push down the trigger to make room for die wheel, seal the case with the top plate, and insert the button on the trigger.
'''

    # Prepare the list of detected objects as a comma-separated string
    objects_list = ', '.join(detected_objects) if detected_objects else 'No objects detected.'

    # Prepare the prompt
    prompt = f"""
    Based on the user's current view of the assembly in the following images and the detected objects ({objects_list}) in the scene, provide the next step in the assembly process.
    """

    # Encode images to base64
    image1_b64 = encode_image(image1)
    image2_b64 = encode_image(image2)
    image3_b64 = encode_image(image3)

    # OpenAI API call
    openai.api_key = os.getenv('OPENAI_API_KEY')
    try:
        API_RESPONSE1 = openai.chat.completions.create(
            model="gpt-4o",  # Corrected model name
            messages=[
            {
                "role": "system",
                "content": assembly_instruction
            },
            {
                "role": "user",
                "content": [ # Changed this line to include content as a list
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image1_b64}" # Changed to jpeg
                        }
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image2_b64}" # Changed to jpeg
                        }
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image3_b64}" # Changed to jpeg
                        }
                    }
                ]
            }
            ]
        )

        instructions = API_RESPONSE1.choices[0].message.content
    except Exception as e:
        instructions = f"Error generating instructions: {str(e)}"

    return instructions

def sample_frames_and_generate_instructions():
    global instructions_text
    global current_objects

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Change index if you have multiple monitors

        while True:
            images = []
            detected_objects_set = set()

            for _ in range(3):
                # Capture the screen
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                images.append(img)

                # Apply object detection on the frame
                results = model.predict(img, verbose=False)

                # Extract object names from the results
                for result in results:
                    for obj in result.boxes:
                        class_name = model.names[int(obj.cls[0])]
                        detected_objects_set.add(class_name)

                time.sleep(3)  # Wait for 3 seconds between captures

            # Aggregate detected objects
            detected_objects = list(detected_objects_set)
            current_objects = detected_objects  # Update the global variable if you choose to display it

            # Generate instructions
            instructions = generate_instructions(images[0], images[1], images[2], detected_objects)

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
    # A cleaner, more professional page
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Assembly King</title>
        <!-- Bootstrap CSS -->
        <link 
            rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" 
            integrity="sha384-MDpi5DInZ8QgjZKT0BS+pMcEc+v4bll73l84I2odZ+0GANC0a3e6aorjj6cZoPsg" 
            crossorigin="anonymous"
        >
        <style>
            body {
                background: #f8f9fa;
                font-family: Arial, sans-serif;
            }
            header {
                background: #343a40;
                color: #fff;
                padding: 20px 0;
                text-align: center;
                margin-bottom: 30px;
            }
            header h1 {
                margin: 0;
                font-size: 2.5rem;
            }
            header p {
                margin: 0;
                font-size: 1rem;
                font-style: italic;
            }
            .instructions-container {
                background: #fff;
                border: 1px solid #dee2e6;
                border-radius: 0.25rem;
                padding: 20px;
                min-height: 150px;
                margin-top: 20px;
                overflow-y: auto;
            }
            .video-container img {
                border: 1px solid #dee2e6;
                border-radius: 0.25rem;
            }
            .footer {
                text-align: center;
                font-size: 0.9rem;
                color: #6c757d;
                margin-top: 30px;
                margin-bottom: 10px;
            }
        </style>
        <script>
            function fetchInstructions() {
                fetch('/instructions')
                    .then(response => response.text())
                    .then(data => {
                        document.getElementById('instructions').innerText = data;
                    });
            }

            function fetchObjects() {
                fetch('/objects')
                    .then(response => response.json())
                    .then(data => {
                        const objects = data.objects;
                        if (objects.length > 0) {
                            document.getElementById('objects').innerText = objects.join(', ');
                        } else {
                            document.getElementById('objects').innerText = 'No objects detected.';
                        }
                    });
            }

            setInterval(fetchInstructions, 9000);  // Fetch instructions every 9 seconds
            setInterval(fetchObjects, 9000);       // Fetch objects every 9 seconds
            window.onload = function() {
                fetchInstructions();  // Fetch instructions on page load
                fetchObjects();       // Fetch objects on page load
            };
        </script>
    </head>
    <body>
        <header>
            <div class="container">
                <h1>Assembly King</h1>
                <p>Be the king of assembly.</p>
            </div>
        </header>

        <div class="container">
            <div class="row">
                <div class="col-md-8 video-container">
                    <h3 class="mb-3">Coach View (AR + Object Detection)</h3>
                    <img src="/video_feed" width="100%" alt="Real-Time Assembly AR View">
                </div>
                <div class="col-md-4">
                    <h3 class="mb-3">Instructions</h3>
                    <div class="instructions-container" id="instructions">
                        <!-- Instructions will be dynamically inserted here -->
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            &copy; 2024 Assembly King by Rongqian Will Chen. All rights reserved.
        </div>
        
        <!-- Optional JavaScript and Bootstrap JS -->
        <script 
            src="https://code.jquery.com/jquery-3.5.1.slim.min.js" 
            integrity="sha384-DfXdBn0Hb9Of/YjHBUN5QFVFQ7VbhfVO80jK2qSWQ5XC92JYMqqx4+pC5aQTBiZO" 
            crossorigin="anonymous"
        ></script>
        <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js" 
            integrity="sha384-LtrjvnR4+o8RM4ut80mc3f+5K5HxbQ7k5jsQXXpXAzvMm9TZM/1o6ID59Nk4Z9CiZ" 
            crossorigin="anonymous"
        ></script>
    </body>
    </html>
    '''

@app.route('/instructions')
def get_instructions():
    global instructions_text
    return instructions_text

@app.route('/objects')
def get_objects():
    global current_objects
    return jsonify({'objects': current_objects})

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