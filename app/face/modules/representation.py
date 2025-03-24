from typing import *

import numpy as np

from face.modules import preprocessing
from face.schemas.FacialRecognition import FacialRecogition
from app.face.models.face_recognition.Facenet import facenet_model

def represent(img: np.ndarray):
    target_size =  facenet_model.input_shape
    
    img = img[:, :, ::-1]
    img = preprocessing.resize_image(img = img,
                                     target_size = (target_size[1], target_size[0]))
    embedding = facenet_model.forward(img)
    return embedding