import os
import pickle
from typing import *

import numpy as np

from .modules import recognition, detection
from .models.OpenCv import opencv_client
from .models.VGGFace import vggface_model
from .helpers import folder_helpers

class Face:
    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        data_path = os.path.join(os.getcwd(), "app", "data")
        stored_data_path = os.path.join(data_path, "data.pkl")
        if os.path.isfile(stored_data_path):
            with open(stored_data_path, "rb") as file:
                return pickle.load(file)
            
        data = {"X": [], "y": [], "X_norm": []}
        for dir in os.scandir(data_path):
            if dir.is_dir():
                backup_file_path = os.path.join(dir.path, "backup.pkl")
                
                try:
                    backup_data = folder_helpers.load_file(backup_file_path)
                    X = backup_data["X"]
                    X_norm = backup_data["X_norm"]
                    y = [dir.name] * len(X)

                    data["X"].extend(X)
                    data["y"].extend(y)
                    data["X_norm"].extend(X_norm)

                except:
                    X = []
                    for file in os.listdir(dir.path):
                        if file.endswith(".jpg"):
                            img_path = os.path.join(dir.path, file)
                            X, _ = detection.extract_embeddings_and_facial_areas(
                                img_path = img_path
                            )

                            X_norm = [np.linalg.norm(embed) for embed in X]
                            y = [dir.name] * len(X)

                            data["X"].extend(X)
                            data["y"].extend(y)
                            data["X_norm"].extend(X_norm)
                    
                    folder_helpers.save_file(
                        objs = {
                            "X": X,
                            "X_norm": X_norm
                        },
                        file_path = os.path.join(dir.path, "backup.pkl")
                    )

        folder_helpers.save_file(
            objs = data,
            file_path = stored_data_path
        )

        return data


    @staticmethod
    def find(
        img_path: Union[str, np.ndarray],
        data: dict,
        distance_metric: str = "cosine",
        threshold: Optional[float] = None
    ):
        return recognition.find(
            img_path = img_path,
            data = data,
            distance_metric = distance_metric,
            threshold = threshold
        )
