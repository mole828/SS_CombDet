'''
web server for detect ball
'''

import base64
import io

from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from Comb import CombineModel

model = CombineModel()

app = FastAPI()
@app.post('/detect')
async def detect(file:UploadFile=File(...)):
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content))
    image = model.draw(image)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    return StreamingResponse(content=buffer, media_type="image/jpeg")


@app.post('/detect_one')
async def detect_one(file:UploadFile=File(...)):
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content))
    crops = model._finder.crops(image)
    if len(crops)!=1: raise ValueError('more or less then one ball')
    crop, conf, space = crops[0]
    buffer = io.BytesIO()
    crop.save(buffer, format='JPEG')
    buffer.seek(0)
    return StreamingResponse(content=buffer, media_type="image/jpeg")


class ColorRequest(BaseModel):
    image: str
    onePieceTaskRecordId: str
    number: str
    physicId: str


@app.post('/ml/color')
async def handle_color_ml(request: ColorRequest):
    color = 'none'
    try:
        image_base64 = request.image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        colors = model.find(image)
        
        match len(colors):
            case 0:
                color = 'white'
            case 1:
                color = colors[0]
            case _:
                raise ValueError(f"More than one ball in camera, len(colors): {len(colors)}")
        
        return JSONResponse(status_code=200, content={
            "code": "0",
            "msg": "请求成功",
            "data": {
                "isRight": True,
                "label": color
            }
        })
    except Exception as error:
        print(error)
        raise HTTPException(status_code=500)
    finally:
        print(f"onePieceTaskRecordId: {request.onePieceTaskRecordId}, physicId: {request.physicId}, number: {request.number}, color: {color}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app= app, host='0.0.0.0')