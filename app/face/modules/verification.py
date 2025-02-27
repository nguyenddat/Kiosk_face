from typing import *

import numpy as np

from ..helpers import model_helpers

def recognize(
        img: np.ndarray,
        data: Dict[AnyStr, Union[np.ndarray, List[str]]],
        threshold: 0.3,
        distance_metric: str = "cosine"
):
    threshold = threshold or model_helpers.find_threshold("VGG-Face", distance_metric)
    X, y = np.array(data["X"]), data["y"]
    img = np.array(img)

    X_norm = np.linalg.norm(X, axis = 1)
    img_norm = np.linalg.norm(img)

    print(y)
    
    distances = np.dot(X, img) / (X_norm * img_norm)
    
    print(distances)
    idx = np.argmax(distances)
    distance, pred = distances[idx], y[idx]

    if distance < threshold:
        return "guest"
    else:
        return pred