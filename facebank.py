import sys
import os
from pathlib import Path
from MTCNN.MTCNN import create_mtcnn_net
from utils.align_trans import *
import numpy as np
from torchvision import transforms as trans
from face_model import MobileFaceNet, l2_norm
import torch
import cv2

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Paths
facebank_path = resource_path('facebank')
pnet_model_path = resource_path('MTCNN/weights/pnet_Weights')
rnet_model_path = resource_path('MTCNN/weights/rnet_Weights')
onet_model_path = resource_path('MTCNN/weights/onet_Weights')
mobile_face_net_path = resource_path('Weights/MobileFace_Net')

def prepare_facebank(model, path=facebank_path, tta=True):
    model.eval()
    embeddings = []
    names = ['']
    data_path = Path(path)

    for doc in data_path.iterdir():
        if doc.is_file():
            continue
        else:
            embs = []
            for files in listdir_nohidden(doc):
                image_path = os.path.join(doc, files)
                img = cv2.imread(image_path)

                if img.shape != (112, 112, 3):
                    bboxes, landmarks = create_mtcnn_net(img, 20, device, 
                        p_model_path=pnet_model_path, 
                        r_model_path=rnet_model_path, 
                        o_model_path=onet_model_path)
                    img = Face_alignment(img, default_square=True, landmarks=landmarks)

                with torch.no_grad():
                    if tta:
                        mirror = cv2.flip(img, 1)
                        emb = model(test_transform(img).to(device).unsqueeze(0))
                        emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))
                        embs.append(l2_norm(emb + emb_mirror))
                    else:
                        embs.append(model(test_transform(img).to(device).unsqueeze(0)))

            if len(embs) == 0:
                continue
            embedding = torch.cat(embs).mean(0, keepdim=True)
            embeddings.append(embedding)
            names.append(doc.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, os.path.join(path, 'facebank.pth'))
    np.save(os.path.join(path, 'names.npy'), names)

    return embeddings, names

def load_facebank(path=facebank_path):
    data_path = Path(path)
    embeddings = torch.load(data_path / 'facebank.pth', weights_only=True)
    names = np.load(data_path / 'names.npy')
    return embeddings, names

if __name__ == '__main__':
    detect_model = MobileFaceNet(512).to(device)  # embedding size is 512 (feature vector)
    detect_model.load_state_dict(
        torch.load(mobile_face_net_path, map_location=lambda storage, loc: storage, weights_only=True))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()

    embeddings, names = prepare_facebank(detect_model, path=facebank_path, tta=True)
    print(embeddings.shape)
    print(names)
