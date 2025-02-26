from typing import *

import numpy as np

from ..helpers import model_helpers

def recognize(
        img: np.ndarray,
        data: Dict[AnyStr, Union[np.ndarray, List[str]]],
        threshold: float,
        distance_metric: str = "cosin"
):
    threshold = threshold or model_helpers.find_threshold("VGG-Face", distance_metric)
    X, y, X_norm = data["X"], data["y"], data["X_norm"]
    img_norm = np.linalg.norm(img)

    distances = np.multiply(
        __x1 = (1 / img_norm) * np.dot(img, X),
        __x2 = X_norm
    )
    
    idx = np.argmax(distances)
    distance, pred = distances[idx], y[idx]

    if distance < threshold:
        return "guest"
    else:
        return pred