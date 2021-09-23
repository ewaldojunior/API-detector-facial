from fastapi import FastAPI
from typing import Dict, List
from detection import DetectFace, DetectEye, DetectMouth
from pydantic import BaseModel
from utils import base64_to_nparray

app = FastAPI()

class Body(BaseModel):
    img: str

@app.post(path="/detect_face")
def detect(body: Body) -> Dict[str, List[List[int]]]:
    detect_face = DetectFace()

    #pre pocessamento
    img = body.img
    img = base64_to_nparray(img)
    detect_face.bgr_to_gray(img)

    #detecção
    res = detect_face.detect(img)

    return {"faces" : res}

@app.post(path='/detect_eye')
def detect(body: Body) -> Dict[str, List[List[int]]]:
    detect_eye = DetectEye()

    #pre pocessamento
    img = body.img
    img = base64_to_nparray(img)
    detect_eye.bgr_to_gray(img)

    #detecção
    res = detect_eye.detect(img)

    return {"eyes" : res}

@app.post(path='/detect_mouth')
def detect(body: Body) -> Dict[str, List[List[int]]]:
    detect_mouth = DetectMouth()

    #pre pocessamento
    img = body.img
    img = base64_to_nparray(img)
    detect_mouth.bgr_to_gray(img)

    #detecção
    res = detect_mouth.detect(img)

    return {"mouth" : res}

#uvicorn app:app --port 8081 --reload