from flask import Flask, render_template_string, Response
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise Exception("Could not open video device")

# Global variables
stop_detection = False
final_counts = {"trees": 0, "people": 0, "faces": 0}
detection_done = False

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
    global stop_detection, final_counts, detection_done

    start_time = time.time()
    duration = 60  # 1 minute

    max_tree_count = 0
    max_people_count = 0
    max_face_count = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        elapsed_time = time.time() - start_time
        remaining_time = max(0, int(duration - elapsed_time))

        if elapsed_time > duration or stop_detection:
            detection_done = True
            break

        tree_count = detect_trees(frame)
        people_count = detect_people(frame)
        face_count = detect_faces(frame)

        total_people_count = people_count + face_count

        max_tree_count = max(max_tree_count, tree_count)
        max_people_count = max(max_people_count, total_people_count)
        max_face_count = max(max_face_count, face_count)

        # Display counts and timer
        cv2.putText(frame, f'Trees: {tree_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'People (incl. faces): {total_people_count}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Faces: {face_count}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'Time left: {remaining_time}s', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    final_counts = {
        "trees": max_tree_count,
        "people": max_people_count,
        "faces": max_face_count
    }

    print("\n[Final Results]")
    print(f"Trees Detected: {max_tree_count}")
    print(f"People Detected (including faces): {max_people_count}")
    print(f"Faces Detected: {max_face_count}")

# === Flask Routes ===

@app.route('/')
def index():
    if detection_done:
        return render_template_string('''
            <html>
            <head><title>Detection Complete</title></head>
            <body style="text-align:center; background:#eee;">
                <h1>Detection Completed!</h1>
                <h2 style="color:green;">Trees Detected: {{ trees }}</h2>
                <h2 style="color:blue;">People (incl. faces): {{ people }}</h2>
                <h2 style="color:orange;">Faces Detected: {{ faces }}</h2>
            
            </body>
            </html>
        ''', trees=final_counts["trees"], people=final_counts["people"], faces=final_counts["faces"])
    else:
        return render_template_string('''
            <html>
            <head>
                <title>Tree, People, and Face Detection</title>
                <script>
                    function stopDetection() {
                        fetch('/stop_detection', { method: 'POST' })
                        .then(response => location.reload());
                    }
                </script>
            </head>
            <body style="background-color: #eee; text-align: center;">
                <h1 style="color: #333;">Live Detection</h1>
                <img src="{{ url_for('video') }}" width="800px" style="border: 5px solid #333;">
                <p>Press the button to stop detection manually:</p>
                <button onclick="stopDetection()">Stop Detection</button>
            </body>
            </html>
        ''')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    global stop_detection
    stop_detection = True
    return '', 204

# === Main ===

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
