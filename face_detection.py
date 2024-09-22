# face_detection.py
import cv2
from mtcnn.mtcnn import MTCNN

def detect_faces(image_path):
    # Load the pre-trained MTCNN face detector
    detector = MTCNN()

    # Read the image
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = detector.detect_faces(rgb_img)

    return faces
