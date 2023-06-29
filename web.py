'''
web server for detect ball
'''

import io

from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

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
async def detect(file:UploadFile=File(...)):
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content))
    crops = model._finder.crops(image)
    if len(crops)!=1: raise ValueError('more or less then one ball')
    crop, conf, space = crops[0]
    buffer = io.BytesIO()
    crop.save(buffer, format='JPEG')
    buffer.seek(0)
    return StreamingResponse(content=buffer, media_type="image/jpeg")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app= app)