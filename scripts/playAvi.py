import cv2
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Display video at specified FPS or scroll frame by frame.")
parser.add_argument('video_path', type=str, help="Path to the .avi video file")
parser.add_argument('--frame_by_frame', action='store_true', help="Enable frame-by-frame mode (press any key to advance)")
args = parser.parse_args()

# Open the video file
cap = cv2.VideoCapture(args.video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video's FPS (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    print("Error: Unable to retrieve FPS from video. Defaulting to 60 FPS.")
    fps = 60

# Calculate the wait time in milliseconds for the desired FPS
frame_duration_ms = int(1000 / fps)

# Loop to read frames and display them
while True:
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("End of video")
        break

    # Display the frame
    cv2.imshow('Video', frame)

    # Handle frame-by-frame or normal playback mode
    if args.frame_by_frame:
        # Wait indefinitely for a key press in frame-by-frame mode
        key = cv2.waitKey(0)
    else:
        # Wait for the calculated frame duration in normal playback mode
        key = cv2.waitKey(frame_duration_ms)

    # Exit if 'q' is pressed
    if key & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
