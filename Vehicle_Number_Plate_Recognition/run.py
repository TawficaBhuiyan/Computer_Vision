import cv2
import argparse
from pipeline import VNPRPipeline

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="VNPR Pipeline with ByteTrack")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", required=True, help="Output video file path")
    args = parser.parse_args()

    # Initialize the core pipeline
    print("[INFO] Loading AI models...")
    pipeline = VNPRPipeline()

    # Open the input video stream
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {args.input}")
        return

    # Extract video properties for the output writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize the VideoWriter to save the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("[INFO] Processing video frames. This may take a while depending on your CPU/GPU...")
    frame_count = 0

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video stream

        # Pass the frame to the pipeline for detection, tracking, and annotation
        annotated_frame = pipeline.process_frame(frame, frame_count)
        
        # Write the annotated frame to the output file
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    # Release memory and close video streams safely
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Task Complete! Annotated video saved to: {args.output}")

if __name__ == "__main__":
    main()