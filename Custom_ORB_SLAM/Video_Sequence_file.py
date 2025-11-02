import cv2
import os


VIDEO_PATH = "C:\\Users\\shanm\\Downloads\\VSLAM_Test.mp4"        
OUTPUT_FOLDER = "C:\\Raja_Docs\\Data"    
FRAME_SKIP = 1                        


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_SKIP == 0:
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved:04d}.png")
        cv2.imwrite(filename, frame)
        saved += 1

    frame_id += 1

cap.release()
print(f"Done! Extracted {saved} frames to: {OUTPUT_FOLDER}")