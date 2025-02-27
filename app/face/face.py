import os
import pickle
from typing import *

import numpy as np

from .modules import recognition, detection
from .helpers import folder_helpers, image_helpers

class Face:
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), "app", "data")
        self.stored_data_path = os.path.join(self.data_path, "data.pkl")
        self.data = self.load_data()

    def load_data(self):
        if os.path.isfile(self.stored_data_path):
            with open(self.stored_data_path, "rb") as file:
                return pickle.load(file)
            
        data = {"X": [], "y": []}
        for dir in os.scandir(self.data_path):
            if dir.is_dir():
                backup_file_path = os.path.join(dir.path, "backup.pkl")
                
                try:
                    backup_data = folder_helpers.load_file(backup_file_path)
                    X = backup_data["X"]
                    y = [dir.name] * len(X)

                    data["X"] += X
                    data["y"] += y

                except:
                    X, y = [], []
                    for file in os.listdir(dir.path):
                        if file.endswith(".jpg"):
                            img_path = os.path.join(dir.path, file)
                            embed, _ = detection.extract_embeddings_and_facial_areas(
                                img_path = img_path,
                                align = True)

                            label = [dir.name] * len(embed)

                            X += embed
                            y += label
                    
                    data["X"] += X
                    data["y"] += y
                    
                    folder_helpers.save_file(
                        objs = {
                            "X": X
                        },
                        file_path = os.path.join(dir.path, "backup.pkl")
                    )


        folder_helpers.save_file(
            objs = data,
            file_path = self.stored_data_path
        )

        return data
    
    def add_data(self, label, imgs):
        data = {"X": [], "X_norm": []}
        save_dir = os.path.join(self.data_path, label)
        backup_file = os.path.join(save_dir, "backup.pkl")
        os.makedirs(save_dir, exist_ok = True)

        for img in imgs:
            embeds, _ = detection.extract_embeddings_and_facial_areas(
                img_path = img,
                align = True
            )

            if len(embeds) != 1:
                print(f"Ảnh không phù hợp để lưu dữ liệu")
                continue

            data["X"].append(embeds)
            data["X_norm"].append(np.linalg.norm(embeds))
        
        folder_helpers.save_file(objs = data, file_path = backup_file)
        self.data["X"].extend(data["X"])
        self.data["X_norm"].extend(data["X_norm"])
        self.data["y"].extend([label] * len(data["X"]))
        
        folder_helpers.save_file(self.data, self.stored_data_path)

    def delete_data(self, label):
        save_dir = os.path.join(self.data_path, label)
        
        y_array = np.array(self.data["y"])
        idx_to_keep = np.where(y_array != label)[0].tolist()
        
        self.data["X"] = [self.data["X"][i] for i in idx_to_keep]
        self.data["X_norm"] = [self.data["X_norm"][i] for i in idx_to_keep]
        self.data["y"] = [self.data["y"][i] for i in idx_to_keep]

        folder_helpers.save_file(self.data, self.stored_data_path)

    def find(self,
        img_path: Union[str, np.ndarray],
        distance_metric: str = "cosine",
        threshold: Optional[float] = 0.3
    ):
        return recognition.find(
            img_path = img_path,
            data = self.data,
            distance_metric = distance_metric,
            threshold = threshold
        )

face_model = Face()