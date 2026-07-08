import cv2
import mediapipe as mp
from mediapipe import solutions as mp_solutions
import time

mp_facedetector = mp_solutions.face_detection
mp_draw = mp_solutions.drawing_utils

cap = cv2.VideoCapture(0)
# ---- Recording setup ----
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_w, frame_h))

with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Camera frame not received. Skipping...")
            continue

        start = time.time()

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results = face_detection.process(image)

        # Convert the RGB image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # Draw the face detection annotations on the image
                mp_draw.draw_detection(image, detection)
                print(id, detection)

                # Get the bounding box information
                bBox = detection.location_data.relative_bounding_box

                h, w, c = image.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime if totalTime > 0 else 0
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Face Detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()