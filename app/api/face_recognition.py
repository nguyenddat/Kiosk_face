import time
import asyncio
from typing import *

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status, WebSocketException

from face.face import face_model
from schemas.face_recognition import RecogniseDataRequest
from services.ConnectionManager import connection_manager

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

@router.post("/api/face-recognition/delete-data")
def delete_data(data):
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không có dữ liệu")

    try:
        face_model.delete_data(data.cccd_id)
    
    except Exception as err:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, err)

@router.post("/api/face-recognition/recognise")
def face_recognition(data: RecogniseDataRequest):
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không có dữ liệu")
    
    try:
        pred = face_model.find(
            img_path = data.img
        )

        return pred

    except Exception as err:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, err)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await connection_manager.connect(websocket)
        t0 = time.time()
        cccd_id = await asyncio.wait_for(
            websocket.receive_text(),
            timeout = 10 - (time.time() - t0)
        )
    
    except asyncio.TimeoutError:
        await websocket.close(status.WS_1008_POLICY_VIOLATION)
        raise WebSocketException(status.WS_1008_POLICY_VIOLATION, "Không nhận được cccd_id sau 10s")
    
    connection_manager.update(websocket, cccd_id)

    while True:
        data = await websocket.receive_text()
        
        try:
            pred = face_model.find(
                img_path = data
            )
            print(pred)
            await connection_manager.send_response({
                "success": True,
                "event": "webcam",
                "payload": pred
            }, websocket)
        
        except Exception as err:
            await connection_manager.send_response({
                "success": False,
                "event": "webcam",
                "payload": [],
                "error": {"code": status.HTTP_500_INTERNAL_SERVER_ERROR, "message": err}
            }, websocket)
        
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket)
        
        except:
            connection_manager.disconnect(websocket)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail = Exception)


