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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app= app)