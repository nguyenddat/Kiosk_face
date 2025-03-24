import numpy as np

from app.face.schemas.FacialRecognition import FacialRecogition
from app.face.models.face_detection.InsightFace import model

class InsightFaceClient(FacialRecogition):
    def __init__(self):
        self.model = model
        self.model_name = "insight_face"
        self.input_shape = (112, 112)
        self.output_shape = 512
        
    def forward(self, img: np.ndarray) -> np.ndarray:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.model.get(img)
        
        if len(faces) == 0:
            return np.zeros((self.output_shape, ))
        
        return faces[0].embedding

# insightface_recognition = InsightFaceClient()
        