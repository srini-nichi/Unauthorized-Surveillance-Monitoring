import sys
import os
from MTCNN.MTCNN import create_mtcnn_net
from utils.align_trans import *
import cv2
from datetime import datetime
import torch
from pathlib import Path

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get user input for image path and person's name
image_path = input("Enter the path to the image: ")
person_name = input("Enter the name of the person: ")

# Load image
image = cv2.imread(image_path)
if image is None:
    sys.exit(f"Error: Could not read image {image_path}")

# Face detection and alignment
bboxes, landmarks = create_mtcnn_net(image, 20, device,
                                     p_model_path='MTCNN/weights/pnet_Weights',
                                     r_model_path='MTCNN/weights/rnet_Weights',
                                     o_model_path='MTCNN/weights/onet_Weights')

if bboxes is None or landmarks is None:
    sys.exit("Error: No face detected in the image.")

# Align the face
warped_face = Face_alignment(image, default_square=True, landmarks=landmarks)
face_bank_path = resource_path('facebank')

# Save the aligned face
data_path = Path(face_bank_path)
save_path = data_path / person_name
if not save_path.exists():
    save_path.mkdir(parents=True)

filename = str(datetime.now())[:-7].replace(":", "-").replace(" ", "-") + '.jpg'
cv2.imwrite(str(save_path / filename), warped_face[0])

print(f"Saved aligned face image to {save_path / filename}")
