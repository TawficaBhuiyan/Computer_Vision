import cv2

video_path = r"F:\Computer_Vision\Multiple_Object_Tracking_with_BoT-SORT_with_ReID\input\input_video.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"FPS: {fps}")
print(f"Total Frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

cap.release()

# FPS: 8.0
# Total Frames: 794
# Duration: 99.25 seconds