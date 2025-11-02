import cv2
from utils import load_images_from_folder
import config

imgs, files = load_images_from_folder(config.DATASET_PATH)
orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES)
video_out = cv2.VideoWriter('kp_detection_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 480))
for img, fn in zip(imgs, files):
    
    kp = orb.detect(img, None)
    
    
    out = cv2.drawKeypoints(
        img, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    out = cv2.resize(out, (1280, 480))

    
    cv2.putText(out, f"{fn} | {len(kp)} keypoints",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    video_out.write(out)
    
    cv2.imshow('ORB Keypoints Sequence', out)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break
video_out.release()
cv2.destroyAllWindows()
