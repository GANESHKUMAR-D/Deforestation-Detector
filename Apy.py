from flask import Flask, render_template_string, Response
import cv2
import numpy as np

app = Flask(__name__)

# Open the default camera
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    raise Exception("Could not open video device")

# === Detection Functions ===

def detect_trees(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tree_contours = []
    min_area = 1000
    min_aspect_ratio = 1.2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio > min_aspect_ratio:
                tree_contours.append(cnt)

    for i, cnt in enumerate(tree_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Tree {i+1}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return len(tree_contours)

def detect_people(frame):
    people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people = people_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Person', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return len(people)

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, 'Face', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return len(faces)

# === Video Stream ===

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect
            tree_count = detect_trees(frame)
            people_count = detect_people(frame)
            face_count = detect_faces(frame)

            total_people = people_count + face_count

            # Display counts
            cv2.putText(frame, f'Trees: {tree_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'People: {total_people}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Faces: {face_count}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Serve frame as multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# === Flask Routes ===

@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Tree, People and Face Detection</title>
            </head>
            <body style="background-color: #eee; text-align: center;">
                <h1 style="color: #333;">Live Detection</h1>
                <img src="{{ url_for('video') }}" width="800px" style="border: 5px solid #333;">
                <p style="color: #666;">Press CTRL+C to stop the server</p>
            </body>
        </html>
    ''')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# === Main App Runner ===

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
