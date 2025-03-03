import os
import shutil
import pickle
from typing import *

import numpy as np
from tqdm import tqdm

from face.modules import recognition, detection
from face.helpers.folder_helpers import load_file, save_file
from face.helpers.image_helpers import base64_to_png

class Face:
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), "app", "data")
        self.stored_data_path = os.path.join(self.data_path, "data.pkl")
        self.data = self.load_data()


    def find(self,
        img_path: Union[str, np.ndarray],
        distance_metric: str = "cosine",
        threshold: Optional[float] = 0.3
    ):
        return recognition.find(
            img_path = img_path,
            data = self.data,
            distance_metric = distance_metric,
            threshold = threshold)


    def extract_embeddings(self, img_path, align):
        embeds, _ = detection.extract_embeddings_and_facial_areas(img_path, align)
        return embeds


    def load_data(self):
        if os.path.isfile(self.stored_data_path):
            return load_file(self.stored_data_path)
        
        # _____________________________________________________________________________
        data = {"X": [], "y": []}
        for dir in tqdm(os.scandir(self.data_path), desc = "Load các thư mục data ảnh"):
            if not dir.is_dir():
                continue
            
            backup_file = os.path.join(dir.path, "backup.pkl")
            if os.path.isfile(backup_file):
                X = load_file(backup_file)
                data["X"].extend(X)
                data["y"].extend([dir.name] * len(X))
                continue
            
        # _____________________________________________________________________________
            X = []
            imgs = [file for file in os.listdir(dir.path) if file.endswith((".png", ".jpg"))]
            if len(imgs) == 0:
                self.delete_data(dir.name)
                continue

            for img in imgs:
                embeds = self.extract_embeddings(
                    os.path.join(dir.path, img),
                    True 
                )

                X.extend(embeds)

            data["X"].extend(X)
            data["y"].extend([dir.name] * len(X))
            save_file(X, backup_file)
        
        save_file(data, self.stored_data_path)
        return data
            
    
    def update_data(self, identity, imgs):
        self.delete_data(identity)
        self.add_data(identity, imgs)


    def add_data(self, identity, imgs):
        dir = os.path.join(self.data_path, identity)
        os.makedirs(dir, exist_ok = True)

        X_add = []
        for idx, img in enumerate(imgs):
            file_path = os.path.join(dir, f"{identity}_{idx}.png")
            base64_to_png(img, file_path)

            embeds = self.extract_embeddings(img, True)
            X_add.extend(embeds)
        
        self.data["X"].extend(X_add)
        self.data["y"].extend([identity] * len(X_add))
        save_file(self.data, self.stored_data_path)
        """
        CODE CẬP NHẬT CƠ SỞ DỮ LIỆU
        """


    def delete_data(self, identity):
        """"""
        dir = os.path.join(self.data_path, identity)
        if not os.path.exists(dir):
            return 
        
        shutil.rmtree(dir)
        X_new, y_new = [], []
        for (x, y) in zip(self.data["X"], self.data["y"]):
            if y != identity:
                X_new.extend([x])
                y_new.extend([y])
        
        self.data = {"X": X_new, "y": y_new}
        """
        CODE CẬP NHẬT CƠ SỞ DỮ LIỆU
        """


face_model = Face()