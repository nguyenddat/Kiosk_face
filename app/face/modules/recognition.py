from typing import *

import numpy as np

from face.modules import detection, verification

def find(
        img_path: Union[str, np.ndarray],
        data: dict,
        distance_metric: str = "cosine",
        threshold: Optional[float] = 0.3
):
    X, y = data["X"], data["y"]

    if len(X) == 0 or len(y) == 0:
        return []
    # -----------------------------------------------------------
    resp_objs = []

    img_embeds, facial_areas = detection.extract_embeddings_and_facial_areas(
        img_path = img_path,
        align = False
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
    


