import os

# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow info and warning messages
import tensorflow as tf

# Suppress TensorFlow specific logs
tf.get_logger().setLevel('ERROR')

import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Load the ASL model
model = load_model('action_rnn1.h5')
actions = np.array(['see you later', 'how are you', 'i am fine', 'nice to meet you',
                    'how can i help you', 'what is your name', 'take care',
                    'where are you from', 'i do not understand', 'can you repeat', 'do you understand'])

# MediaPipe setup
mp_holistic = mp.solutions.holistic

# Initialize a buffer to hold sequence frames
sequence_buffer = []

def mediapipe_detection(image, model):
    """Detect landmarks in the given image using MediaPipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Run prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results for pose, face, and hands."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_call')
def video_call():
    """Render the video call page."""
    return render_template('video_call.html')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame data from the client."""
    # Decode the base64 frame data
    img_data = base64.b64decode(data['frame'].split(',')[1])
    img = np.array(Image.open(BytesIO(img_data)))

    # Perform MediaPipe detection
    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        image, results = mediapipe_detection(img, holistic)
        keypoints = extract_keypoints(results)

        # Append keypoints to sequence buffer
        sequence_buffer.append(keypoints)
        if len(sequence_buffer) > 30:
            sequence_buffer.pop(0)  # Keep buffer size constant

        # Run ASL prediction if sequence is long enough
        if len(sequence_buffer) == 30:
            sequence = np.array(sequence_buffer)
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            action = actions[np.argmax(res)]
            emit('prediction', {'action': action})
        else:
            emit('prediction', {'action': 'Detecting...'})

if __name__ == '__main__':
    app.run(debug=True)
