from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from predict import Predict


app = FastAPI()

class WholePipelineModel:
    def __init__(self):
        # Здесь должен быть код загрузки модели
        pass

    def predict(self, image_bytes):
        # Здесь должен быть код предобработки и выполнения инференса
        return {"prob": 0.8, "verdict": "OK"}  # Пример результата инференса



whole_pipeline_model = WholePipelineModel()


@app.post('/whole_pipeline')
async def whole_pipeline_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = whole_pipeline_model.predict(image_bytes)
    return result


# class OcrLogRegModel:
#     def __init__(self):
#         # Здесь должен быть код загрузки модели
#         pass
#
#     def predict(self, image_bytes):
#         # Здесь должен быть код предобработки и выполнения инференса
#         return {"text": "Hello World"}  # Пример результата инференса
#
# class YoloDetectionModel:
#     def __init__(self):
#         # Здесь должен быть код загрузки модели
#         pass
#
#     def predict(self, image_bytes):
#         # Здесь должен быть код предобработки и выполнения инференса
#         return {"objects": ["person", "car"]}  # Пример результата инференса
# ocr_log_reg_model = OcrLogRegModel()
# yolo_detection_model = YoloDetectionModel()
# @app.post('/ocr_log_reg')
# async def ocr_log_reg_endpoint(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     result = ocr_log_reg_model.predict(image_bytes)
#     return result
#
# @app.post('/yolo_detection')
# async def yolo_detection_endpoint(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     result = yolo_detection_model.predict(image_bytes)
#     return result
