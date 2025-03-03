from insightface.app import FaceAnalysis

from face.schemas.Detector import *

class InsightFaceClient(Detector):
    def __init__(self):
        self.model = FaceAnalysis("buffalo_l")
        self.model.prepare(ctx_id = 0, det_size = (640, 640))
    
    def detect_faces(self, img) -> List[FacialAreaRegion]:
        faces = self.model.get(img)
        detected_faces = []

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            facial_region = FacialAreaRegion(
                x = x1, y = y1, w = x2 - x1, h = y2 - y1,
                left_eye = tuple(map(int, face.kps[1])),
                right_eye = tuple(map(int, face.kps[0])),
                nose = tuple(map(int, face.kps[2])),
                left_mouth = tuple(map(int, face.kps[4])),
                right_mouth = tuple(map(int, face.kps[3])),
                confidence = float(face.det_score)
            )

            detected_faces.append(facial_region)

        return detected_faces

insightface_client = InsightFaceClient()