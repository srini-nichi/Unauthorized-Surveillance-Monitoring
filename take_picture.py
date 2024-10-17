import sys
import os
from MTCNN.MTCNN import create_mtcnn_net
from utils.align_trans import *
import cv2
from datetime import datetime
import torch
from pathlib import Path

# Function to get resource path for PyInstaller or development environment
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prompt the user to input the name of the person
person_name = input("Enter the name of the person: ")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        frame_text = cv2.putText(frame, 'Press t to take a picture, q to quit...', (10, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("My Capture", frame_text)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        p = frame

        # Create the folder for the person inside facebank if it doesn't exist
        data_path = Path('facebank')
        save_path = data_path / person_name
        if not save_path.exists():
            save_path.mkdir(parents=True)

        try:
            p_model_path = resource_path('MTCNN/weights/pnet_Weights')
            r_model_path = resource_path('MTCNN/weights/rnet_Weights')
            o_model_path = resource_path('MTCNN/weights/onet_Weights')

            # Detect face and align
            bboxes, landmarks = create_mtcnn_net(p, 20, device, 
                                                 p_model_path=p_model_path, 
                                                 r_model_path=r_model_path, 
                                                 o_model_path=o_model_path)
            if bboxes is not None:
                warped_face = Face_alignment(p, default_square=True, landmarks=landmarks)
                filename = str(datetime.now())[:-7].replace(":", "-").replace(" ", "-") + '.jpg'
                cv2.imwrite(str(save_path / filename), warped_face[0])
                print(f"Saved face image to {save_path / filename}")
            else:
                print("No face detected. Please try again.")
        except Exception as e:
            print(f"Error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('My Capture', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
