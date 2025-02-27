from typing import *

from pydantic import BaseModel

class RecogniseDataRequest(BaseModel):
    img: str