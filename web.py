'''
web server for detect ball
'''

import base64
import io

from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse

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


@app.post('/ml/color')
async def classify_color(data: ImageData):
    try:
        # Convert base64 image to a PIL Image
        image_base64 = data.image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # # Convert image to a NumPy array and normalize the pixel values
        # image_array = np.array(image) / 255.0
        
        # # Predict the class of the image
        # result = model.predict(np.array([image_array]))
        # predictions = [
        #     {"label": dict_labels[str(index)], "prob": float(value)}
        #     for index, value in enumerate(result[0])
        # ]
        # predictions = sorted(predictions, key=lambda x: x['prob'], reverse=True)
        # item = predictions[0]
        # is_right = True if item['prob'] >= 0.5 else False
        colors = model.find(image)
        if len(colors) != 1:
            return {
                'code': '0',
                'msg': '请求成功',
                'data': {
                    'isRight': False,
                    'label': 'unknow'
                },
            }

        # Post the result to a specific URL
        # Make sure to replace 'ssBasicUrl' with the actual URL
        # ssBasicUrl = "http://example.com"
        # url = f"{ssBasicUrl}/api/master/ml/logMlResult"
        # headers = {'Content-Type': 'application/json', 'appid': 'basic'}
        # data = {
        #     "image": image_base64,
        #     "onePieceTaskRecordId": data.onePieceTaskRecordId,
        #     "number": data.number,
        #     "physicId": data.physicId,
        #     "predictions": predictions,
        #     "closestPrediction": item,
        #     "isRight": is_right
        # }
        # response = requests.post(url, headers=headers, json=data)
        
        # Return the response
        return JSONResponse(status_code=200, content={
            "code": "0",
            "msg": "请求成功",
            "data": {
                "isRight": True,
                "label": colors[0]
            }
        })
    except Exception as error:
        print(error)
        raise HTTPException(status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app= app)