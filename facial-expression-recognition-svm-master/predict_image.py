import os
import cv2
import run_clf as rclf

IMAGE_DIR = "test_images"


for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    print("Đang test:", img_name)

    frame = cv2.imread(img_path)
    if frame is None:
        continue

    frame = rclf.predict_frame(frame)
    cv2.imshow("Emotion - Image", frame)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
