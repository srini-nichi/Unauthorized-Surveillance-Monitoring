import sys
import os
import argparse
import torch
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import *
from utils.align_trans import *
from MTCNN.MTCNN import create_mtcnn_net
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank, prepare_facebank
import cv2
import time
import queue
import threading
import warnings
import pygame
import logging

# Set up logging
log_file_path = "unauthorized-monitoring.log"
logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Suppress warnings and logs from other libraries
os.environ["SDL_HINT_LOGGING_LEVEL"] = "0"  # Suppress SDL logs
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Log startup message
logging.info("Starting the face recognition application.")

# Initialize pygame for sound playback
pygame.mixer.init()

# Load buzzer sound file from the resources
buzzer_sound = pygame.mixer.Sound(resource_path("utils/warning.wav"))

def resource_path(relative_path):
    """ 
    Get absolute path to resource, works for development and for PyInstaller builds.
    
    Parameters:
        relative_path (str): Path relative to the script or executable.

    Returns:
        str: Absolute path to the resource.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



def resource_path_facebank(relative_path):
    """ 
    Get absolute path to resource specific to the facebank, 
    suitable for both development and PyInstaller builds.
    
    Parameters:
        relative_path (str): Path relative to the facebank directory.

    Returns:
        str: Absolute path to the resource.
    """
    # This should point to the directory of the main executable
    base_path = os.path.dirname(os.path.abspath(sys.argv[0]))  # Main executable directory
    return os.path.join(base_path, relative_path)

def resize_image(img, scale):
    """ 
    Resize an image by a scale factor.
    
    Parameters:
        img (numpy.ndarray): The image to resize.
        scale (float): The scale factor for resizing.

    Returns:
        numpy.ndarray: The resized image.
    """
    height, width, channel = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    return img_resized

# Initialize queues for frame capture and processing
frame_queue = queue.Queue(maxsize=4)
result_queue = queue.Queue(maxsize=4)
stop_threads = False

# Thread function to capture frames from the camera
def capture_frames(cap):
    """ 
    Capture frames from the camera and put them into the frame queue. 
    If the quality is 2k, compress it 720p.
    
    Parameters:
        cap (cv2.VideoCapture): Video capture object for the camera.
    """
    while True:
        try:
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret: # Check if the frame was captured successfully
                logging.warning("Warning: Failed to read frame. Retrying...")
                time.sleep(0.1) # Wait before retrying
                continue

             # Resize the frame for 2k quality camera to 720p HD
            if frame.shape[1] == 2560 and frame.shape[0] == 1440:
                frame_resized = resize_image(frame, 0.5)
            else:
                frame_resized = frame

            # Add the resized frame to the queue
            if not frame_queue.full():
                frame_queue.put(frame_resized)
            else:
                try:
                    frame_queue.get_nowait() # Remove the oldest frame if queue is full
                    frame_queue.put(frame_resized) # Add the new frame
                except queue.Empty:
                    pass # Ignore if the queue is empty
        except Exception as e:
            logging.error(f"Exception in capture_frames: {e}")
            time.sleep(0.5) # Wait before retrying

# Thread function to process frames from the queue
def process_frames(args, device, detect_model, targets, names):
    """ 
    Process frames from the queue and detect faces and embeddings.
    
    Parameters:
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Device to perform computations on (CPU/GPU).
        detect_model (torch.nn.Module): Face detection model.
        targets (torch.Tensor): Precomputed embeddings of known faces.
        names (list): Names associated with the known faces.
    """
    prev_bboxes, prev_landmarks, prev_results, prev_score_100 = [], [], [], []
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get() # Get the next frame from the queue
            try:
                input = resize_image(frame, args.scale) # Resize input frame for processing
                bboxes, landmarks = create_mtcnn_net( 
                    input, args.mini_face, device, 
                    p_model_path=resource_path('MTCNN/weights/pnet_Weights'),
                    r_model_path=resource_path('MTCNN/weights/rnet_Weights'),
                    o_model_path=resource_path('MTCNN/weights/onet_Weights')
                )

                if stop_threads:
                    break # Exit loop if stop signal received

                if bboxes is not None and len(bboxes) > 0:
                    # Scale bounding boxes and landmarks back to original frame size
                    bboxes = bboxes / args.scale
                    landmarks = landmarks / args.scale

                    # Align faces using detected landmarks
                    faces = Face_alignment(frame, default_square=True, landmarks=landmarks)
                    embs = []
                    test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])

                    # Extract embeddings for each face
                    for img in faces:
                        emb = detect_model(test_transform(img).to(device).unsqueeze(0))
                        embs.append(emb)

                    if embs:
                        # Compare detected embeddings with targets in the facebank
                        source_embs = torch.cat(embs)
                        diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0)
                        dist = torch.sum(torch.pow(diff, 2), dim=1)
                        minimum, min_idx = torch.min(dist, dim=1)
                        min_idx[minimum > ((args.threshold-156)/(-80))] = -1
                        results = min_idx
                        score_100 = torch.clamp(minimum * -80 + 156, 0, 100)
                        prev_bboxes, prev_landmarks, prev_results, prev_score_100 = bboxes, landmarks, results, score_100

                else:
                    results = []
                    score_100 = []
                    bboxes = []
                # Store the results in the result queue for display
                prev_bboxes, prev_landmarks, prev_results, prev_score_100 = bboxes, landmarks, results, score_100
                result_queue.put((frame, prev_bboxes, prev_landmarks, prev_results, prev_score_100))

            except Exception as e:
                logging.error('Detection error: %s', e)
                result_queue.put((frame, [], [], [], [])) # Push empty results if error occurs
                if stop_threads:
                    break

if __name__ == '__main__':
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Face detection demo')
    parser.add_argument('-th', '--threshold', help='Threshold score to decide identical faces', default=60, type=float)
    parser.add_argument("-u", "--update", help="Whether to perform update of the facebank", action="store_true", default=False)
    parser.add_argument("-tta", "--tta", help="Whether to use test time augmentation", action="store_true", default=False)
    parser.add_argument("-c", "--score", help="Whether to show the confidence score", action="store_true", default=True)
    parser.add_argument("--scale", help="Input frame scale to adjust the speed", default=0.5, type=float)
    parser.add_argument('--mini_face', help="Minimum face size to be detected", default=40, type=int)
    parser.add_argument('--process_every', help="Process every nth frame", default=3, type=int)
    parser.add_argument("-ci", "--camera_index", help="Camera index or RTSP URL (default is 0)", default=0, type=str)
   
    args = parser.parse_args() # Parse command line arguments
    logging.info("Parsed arguments and set up configurations.")

    def get_settings_file_path():
        """ 
        Get the path for saving camera settings based on the execution context (script or executable).
        
        Returns:
            str: Full path to the camera settings file.
        """
        # Check if running as a standalone executable
        if getattr(sys, 'frozen', False):
            # If the script is running as an executable, use the temporary folder
            current_directory = os.path.dirname(sys.executable)
        else:
            # If running as a script, use the script's directory
            current_directory = os.path.dirname(os.path.abspath(__file__))
        
        return os.path.join(current_directory, "camera_settings.txt")

    def save_camera_setting(setting):
        with open(get_settings_file_path(), "w") as f:
            f.write(setting)

    def load_camera_setting():
        settings_file = get_settings_file_path()
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                return f.read().strip()
        return None

    def get_camera_source():
        saved_source = load_camera_setting()
        if saved_source:
            print(f"Press Enter to use the last time used camera index/RTSP link: {saved_source}")
            source = input(f"Enter camera index or RTSP link (default index is 0): ").strip()
            source = source if source else saved_source  # Use saved setting if no input is provided
        else:
            source = input("Enter camera index or RTSP link (default index is 0): ").strip() or "0"

        # Save the new choice for future use
        save_camera_setting(source)  

        # Convert the source to an integer if it's a camera index
        try:
            return int(source)  # Return as integer if valid camera index
        except ValueError:
            return source  # Return as string if it’s an RTSP link

    # Get the camera source
    camera_source = get_camera_source()

    # Set the camera index or RTSP link
    camera_index = camera_source

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detect_model = MobileFaceNet(512).to(device)
    weight_path = resource_path('Weights/MobileFace_Net')
    try:
        detect_model.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage, weights_only=True))
        print('MobileFaceNet face detection model loaded')
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

    detect_model.eval()
    face_bank_path = resource_path_facebank('facebank')

    if args.update:
        targets, names = prepare_facebank(detect_model, path=face_bank_path, tta=args.tta)
        print('Facebank updated')
    else:
        targets, names = load_facebank(path=face_bank_path)
        print('Facebank loaded')

    cv2.namedWindow('Unauthorized Monitoring', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unauthorized Monitoring', 640, 480)
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    aoi = None
    define_aoi = False
    warning_displayed = False
    image_captured = False

    # Start threads for frame capture and processing
    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    capture_thread.daemon = True
    capture_thread.start()

    process_thread = threading.Thread(target=process_frames, args=(args, device, detect_model, targets, names))
    process_thread.daemon = True
    process_thread.start()

    prev_frame_time = 0
    fps_array = []
    fps_array_size = 30

    warning_message = ""

    # Variables to manage the blinking warning message
    blink_state = False
    blink_interval = 2  # Change this value to control the blinking speed
    blink_counter = 0

    while True:
        if not result_queue.empty():
            frame, bboxes, landmarks, results, score_100 = result_queue.get()

            if define_aoi:
                aoi = cv2.selectROI("Unauthorized Monitoring", frame, showCrosshair=True, fromCenter=False)
                define_aoi = False

            if bboxes is None or len(bboxes) == 0:
                bboxes, landmarks, results, score_100 = [], [], [], []

            new_frame_time = time.time()
            time_diff = new_frame_time - prev_frame_time
            if time_diff > 0:
                fps = 1 / time_diff
                fps_array.append(fps)
                if len(fps_array) > fps_array_size:
                    fps_array.pop(0)
                avg_fps = int(sum(fps_array) / len(fps_array))
            else:
                avg_fps = 0 if not fps_array else int(fps_array[-1])
            prev_frame_time = new_frame_time

            if aoi is not None:
                x, y, w, h = aoi
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

            if bboxes is not None:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                font_path = resource_path("utils/Roboto-Regular.ttf")
                font = ImageFont.truetype(font_path, 30)

                for i, b in enumerate(bboxes):
                    if results[i] == -1:  # Unknown face
                        # Draw a closed triangle for unknown faces
                        triangle = [(b[0], b[3]), (b[2], b[3]), ((b[0] + b[2]) / 2, b[1])]
                        draw.polygon(triangle, outline='red', fill=None, width=5)
                        # Draw the 'Unknown' text below the triangle
                        text_position = ((b[0] + b[2]) / 2, b[3] + 5)
                        draw.text(text_position, 'Unknown', fill=(255, 0, 0), font=font)

                        # Check if the unknown face is within the AOI
                        if aoi is not None and (b[0] >= x and b[2] <= x + w and b[1] >= y and b[3] <= y + h):
                            if not warning_displayed:
                                warning_message = "Warning!!! Unknown person detected in Unauthorized Area!"
                                print("Suspicion detected!")
                                buzzer_sound.play(loops=-1)
                                warning_displayed = True

                    else:  # Recognized face
                        # Draw a green rectangle for recognized faces
                        draw.rectangle([b[0], b[1], b[2], b[3]], outline='green', width=5)
                        name_position = (b[0], b[1] - 30)
                        name_text = names[results[i] + 1] if args.score else names[results[i]]
                        draw.text(name_position, name_text, fill=(255, 255, 0) if args.score else (0, 255, 0), font=font)

                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Reset warning if no unknown faces are detected in the AOI
            if warning_displayed:
                unknown_faces_in_aoi = any(
                    (results[i] == -1 and
                    (bbox[0] >= x and bbox[0] <= x + w) and
                    (bbox[1] >= y and bbox[1] <= y + h))
                    for i, bbox in enumerate(bboxes))
                if not unknown_faces_in_aoi:
                    buzzer_sound.stop()
                    warning_displayed = False
                    warning_message = "" 

            # Update the blink state for the warning message
            blink_counter += 1
            if blink_counter >= blink_interval:
                blink_state = not blink_state
                blink_counter = 0

            # Display the warning message if the blink state is True
            if warning_message and blink_state:
                cv2.putText(frame, warning_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Unauthorized Monitoring', frame)

        key = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Unauthorized Monitoring', cv2.WND_PROP_VISIBLE) < 1:
            stop_threads = True
            break
        elif key == ord('m'):
            
            define_aoi = True
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    sys.exit()