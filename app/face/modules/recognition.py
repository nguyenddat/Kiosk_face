from typing import *

import numpy as np

from . import detection, verification

def find(
        img_path: Union[str, np.ndarray],
        data: dict,
        distance_metric: str = "cosine",
        threshold: Optional[float] = None
):
    X, y, X_norm = data["X"], data["y"], data["X_norm"]

    if len(X) == 0 or len(y) == 0 or len(X_norm) == 0:
        return []
    # -----------------------------------------------------------
    resp_objs = []

    source_objs = detection.extract_faces(img_path)
    for source_obj in source_objs:
        img_embeds, facial_areas = detection.extract_embeddings_and_facial_areas(
            source_obj["img"]
        )

        for img_embed, facial_area in zip(img_embeds, facial_areas):
            resp_obj = verification.recognize(
                img = img_embed,
                data = data,
                threshold = threshold,
                distance_metric = distance_metric
            )

            resp_objs.append({
                "prediction": resp_obj,
                "facial_area": facial_area
            })
    
    return resp_objs
    


