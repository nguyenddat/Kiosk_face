import json
from typing import List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.__active_connections: dict[WebSocket: str] = {}
    
    def get_active_connections(self):
        return self.__active_connections
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()

    def update(self, websocket: WebSocket, id_cccd: str):
        self.__active_connections.update({websocket: id_cccd})
        print(f"Cập nhật: Thêm {websocket}: {id_cccd}\n    Có {len(self.__active_connections)} máy đang kết nối")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.__active_connections.keys():
            self.__active_connections.pop(websocket)
        print(f"Cập nhật: Xóa {websocket}\n    Có {len(self.__active_connections)} máy đang kết nối")
        
    async def send_response(self, response: dict, websocket: WebSocket):
        message = json.dumps(response)
        await websocket.send_text(message)

    async def broadcast_all(self, response: dict):
        for connection in self.__active_connections:
            await connection.send_text(json.dumps(response))
            
    async def broadcast_some(self, response: dict, websocket_list: List[WebSocket]):
        for connection in websocket_list:
            await connection.send_text(json.dumps(response))
            
connection_manager = ConnectionManager()