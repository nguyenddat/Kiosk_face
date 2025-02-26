from typing import *

import cv2
import numpy as np

from . import preprocessing, representation
from ..models.OpenCv import opencv_client
from ..schemas.Detector import *
from ..helpers import image_helpers


def extract_embeddings_and_facial_areas(
        img_path: Union[str, np.ndarray]
):
    embeddings = []
    facial_areas = []

    resp_objs = extract_faces(img_path = img_path, align = align)
    for resp_obj in resp_objs:
        current_img = resp_obj["img"]

        img_embed = representation.represent(img = current_img)
        embeddings.append(img_embed)
        facial_areas.append(resp_obj["facial_area"])
    
    return embeddings, facial_areas

def extract_faces(img_path: Union[str, np.ndarray], align: bool):
    img, img_name = image_helpers.load_image(img_path)
    if img is None:
        raise ValueError(f"Exception while loading image: {img_name}")

    h, w = img.shape[:2]
    base_region = FacialAreaRegion(
        x = 0, y = 0, w = w, h = h,
        confidence = 0
    )

    face_objs = detect_faces(img = img, align = align)
    if len(face_objs) == 0:
        face_objs = [DetectedFace(img = img, facial_area = base_region, confidence = 0)]
    
    # --------------------------------------------------
    resp_objs = []
    for face_obj in face_objs:
        current_img = (face_obj.img) / 255
        current_region = face_obj.facial_area

        if (current_img.shape[0] == 0 or current_img.shape[1] == 0): continue

        x = max(0, int(current_region.x))
        y = max(0, int(current_region.y))
        w = min(w - x - 1, int(current_region.w))
        h = min(h - y - 1, int(current_region.h))

        facial_area = {
            "x": x, "y": y, "w": w, "h": h,
            "left_eye": current_region.left_eye,
            "right_eye": current_region.right_eye
        }

        resp_objs.append({
            "img": current_img,
            "facial_area": facial_area,
            "confidence": round(float(current_region.confidence or 0), 2)
        })
    
    return resp_objs

def detect_faces(img, align) -> List[DetectedFace]:
    h, w = img.shape[:2]
    h_border = int(0.5 * h)
    w_border = int(0.5 * w)

    img = cv2.copyMakeBorder(
        src = img,
        top = h_border,
        bottom = h_border,
        left = w_border,
        right = w_border,
        borderType = cv2.BORDER_CONSTANCE,
        value = [0, 0, 0]
    )

    facial_areas = opencv_client.detect_faces(img)

    return [
        extract_face(
            img = img,
            facial_area = facial_area,
            width_border = w_border,
            height_border = h_border,
            align = align
        ) for facial_area in facial_areas
    ]

def extract_face(
        img: np.ndarray, 
        facial_area: FacialAreaRegion, 
        width_border: int, 
        height_border: int,
        align: bool
) -> DetectedFace:
    
    x, y, w, h = facial_area.x, facial_area.y, facial_area.w, facial_area.h
    left_eye, right_eye = facial_area.left_eye, facial_area.right_eye
    nose = facial_area.nose
    left_mouth, right_mouth = facial_area.left_mouth, facial_area.right_mouth
    confidence = facial_area.confidence 

    detected_face = img[
        int(y): int(y + h),
        int(x): int(x + w)
    ]

    if left_eye is not None and right_eye is not None and align:
        sub_img, relative_x, relative_y = preprocessing.extract_sub_img(
            img = img,
            facial_area = (x, y, w, h)
        )

        aligned_sub_img, angle = preprocessing.align_img_with_eyes(
            img = img,
            left_eye = left_eye,
            right_eye = right_eye
        )

        rotated_x1, rotated_y1, rotated_x2, rotated_y2 = preprocessing.project_facial_area(
            facial_area = (
                relative_x,
                relative_y,
                relative_x + w,
                relative_y + h
            ),
            angle = angle,
            size = (sub_img.shape[0], sub_img.shape[1])
        )

        detected_face = aligned_sub_img[
            int(rotated_y1): int(rotated_y2),
            int(rotated_x1): int(rotated_x2)
        ]

        x = x - width_border
        y = y - height_border

        if left_eye is not None:
            left_eye = (left_eye[0] - width_border, left_eye[1] - height_border)            
        if right_eye is not None:
            right_eye = (right_eye[0] - width_border, right_eye[1] - height_border)
    
    return DetectedFace(
        img = detected_face,
        facial_area = FacialAreaRegion(
            x = x, y = y, w = w, h = h,
            confidence = confidence,
            left_eye = left_eye,
            right_eye = right_eye,
            nose = nose,
            left_mouth = left_mouth,
            right_mouth = right_mouth
        ),
        confidence = confidence or 0
    )
