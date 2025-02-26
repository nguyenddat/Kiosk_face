import os
import time
import shutil
from typing import *

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status, WebSocketException

from face.face import face_model


router = APIRouter()

@router.post("/api/face-recogntion/add-data")
def add_data(data):
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không có dữ liệu")
    
    imgs = data.b64_img
    personal_data = data.cccd
    role = data.role
    personal_id = personal_data.get("Identity Code")

    if not personal_id or not role:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Thiếu thông tin định danh")

    try:
        face_model.add_data(label = personal_id, imgs = imgs)

    except Exception as err:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, err)

@router.post("/api/face-recognition/recognise")
def face_recognition(data):
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không có dữ liệu")
    
    try:
        pred = face_model.find(
            img_path = data
        )

        return pred

    except Exception as err:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, err)
