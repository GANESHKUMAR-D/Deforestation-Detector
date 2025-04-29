import cv2
import numpy as np
import time

def count_trees_faces_and_people_camera_one_minute():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Counting trees, faces, and people for 1 minute...")

    # Set the start time
    start_time = time.time()
    end_time = start_time + 60  # 60 seconds = 1 minute

    max_tree_count = 0
    max_people_count = 0
    max_face_count = 0
    final_result_image = None

    while time.time() < end_time:
        remaining_time = int(end_time - time.time())
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Detect trees, people, and faces
        tree_frame, tree_count = detect_trees(frame)
        people_frame, people_count = detect_people(frame)
        face_frame, face_count = detect_faces(frame)

        # Combine people and face counts
        total_people_count = people_count + face_count

        # Update maximum counts
        max_tree_count = max(max_tree_count, tree_count)
        max_people_count = max(max_people_count, total_people_count)
        max_face_count = max(max_face_count, face_count)
        final_result_image = tree_frame.copy()

        # Display counts and remaining time on the frame
        cv2.putText(final_result_image, f"Trees: {tree_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(final_result_image, f"People: {total_people_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(final_result_image, f"Faces: {face_count}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(final_result_image, f"Time left: {remaining_time}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Detection', final_result_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Print final counts
    print(f"Final tree count: {max_tree_count}")
    print(f"Final people count (including faces): {max_people_count}")
    print(f"Final face count: {max_face_count}")


def detect_trees(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tree_contours = []
    min_area = 1000
    min_height_ratio = 1.5

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio > min_height_ratio:
                tree_contours.append(contour)

    result = frame.copy()
    for i, contour in enumerate(tree_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"Tree {i+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result, len(tree_contours)


def detect_people(frame):
    people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people = people_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    result = frame.copy()
    for (x, y, w, h) in people:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result, len(people)


def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    result = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(result, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return result, len(faces)


if __name__ == "__main__":
    count_trees_faces_and_people_camera_one_minute()
