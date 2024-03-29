import cv2 
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
import asyncio
import uvicorn
from openvino.runtime import Core
from paddleocr import PaddleOCR
from get_plate import get_plate
from read_plate import read_plate, read_plate_no_detection
import base64

# Load the model 
ie = Core()
model = ie.read_model(model="./plate_detection_model/saved_model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer_ir = next(iter(compiled_model.inputs))

ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log = False, use_gpu=True) # need to run only once to download and load model into memory


# Fast API
app = FastAPI()

@app.get("/")
def index():
    return {"name" : "giabao"}

@app.post("/plate/")
async def plate(name_cam: str = Form(""), image: str = Form("")):
    try:
        if image == "":
            return jsonable_encoder({
                "code": 201,
                "error_code": 1,
                "msg": "Missing Input Image"
            })
        
        # check image
        contents = base64.b64decode(image)
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonable_encoder({
                "code": 201,
                "error_code": 2,
                "msg": "Input is not an image"
            })

        # get plate
        plate = get_plate(img, compiled_model, input_layer_ir)
        if plate is None: 
            return jsonable_encoder({
                "code": 201,
                "error_code": 3,
                "msg": "Plate not found"
            }) 

        plate_text = read_plate_no_detection(plate, ocr)
        if plate_text is None:
            return jsonable_encoder({
                "code": 201,
                "error_code": 4,
                "msg": "Plate not recognize" # NEED MODIFY
            })
            
        print(plate_text)

        return jsonable_encoder({
            "code": 200,
            'name_cam': name_cam,
            "plate_text": plate_text,
            "msg": "success"
        }) 

    except Exception as e:
        print(e)
        return jsonable_encoder({
                "code": 201,
                "error_code": 0,
                "msg": str(e)
            })

@app.post("/plate_file/")
async def plate(name_cam: str = Form(""), image: UploadFile = File(None)):
    try:
        if image == None:
            return jsonable_encoder({
                "code": 201,
                "error_code": 1,
                "msg": "Missing Input Image"
            })

        contents = await asyncio.wait_for(image.read(), timeout=1) 
        if(str(contents) =="b''"):
            return jsonable_encoder({
                "code": 201,
                "error_code": 2,
                "msg": "Not found file"
            })
        
        # check image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonable_encoder({
                "code": 201,
                "error_code": 3,
                "msg": "Input is not an image"
            })

        # get plate
        plate = get_plate(img, compiled_model, input_layer_ir)
        if plate is None: 
            return jsonable_encoder({
                "code": 201,
                "error_code": 3,
                "msg": "Plate not found"
            }) 

        plate_text = read_plate(plate, ocr)
        if plate_text is None:
            return jsonable_encoder({
                "code": 201,
                "error_code": 4,
                "msg": "Plate not recognize" # NEED MODIFY
            })
            
        print(plate_text)

        return jsonable_encoder({
            "code": 200,
            'name_cam': name_cam,
            "plate_text": plate_text,
            "msg": "success"
        }) 

    except Exception as e:
        print(e)
        return jsonable_encoder({
                "code": 201,
                "error_code": 0,
                "msg": str(e)
            })

if __name__ == "__main__":
    # run API
    uvicorn.run('app:app', host="127.0.0.1", port=8400, reload=True)