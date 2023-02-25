import logging

from src.const.global_map import RESOURCE_MAP
from src.const import const_map as CONST_MAP
from src.api_endpoint.add_api import api_log_aischema
from src.utils.basemodel import app_schemas as schemas
from src.utils.basemodel.response_schemas import create_response, ResponseModel
from src.api_endpoint.load import df
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pipeline import run
ai_logger = logging.getLogger("ai_logger")
app_logger = logging.getLogger("app_logger")

app = RESOURCE_MAP["fastapi_app"]

@app.post("/detect", response_model=ResponseModel)
# @api_log_aischema
async def predict(input_map: schemas.IDSSchema) -> ResponseModel:
    model = tf.keras.models.load_model('resource/my_model')
    data = []
    for i in range(len(input_map.duration)):
        pre_data = []
        for f in CONST_MAP.input:
            pre_data.append(1.0*(input_map.dict()[f][i]-df[f].min())/(df[f].max()-df[f].min()))
        data.append(pre_data)
        
    data = np.array(data)
    Y_train_pred = model.predict(data)
    attack_index = np.where(Y_train_pred[0] == np.amax(Y_train_pred[0]))[0][0]

    return create_response(status_code=200, content={ 
                                                     "attack": CONST_MAP.attack_type[attack_index]
                                                     })


@app.post("/train_model", response_model=ResponseModel)
# @api_log_aischema
async def predict(input_map: schemas.Datasetchema) -> ResponseModel:
    # res = "Url None"
    bucket_name = "data"
    object_name = "data.npy"
    try:
        client = Minio(
            "13.215.50.232:9000",
            access_key= "uit",
            secret_key= "uituituit",
            secure=False
        )
        client.fget_object(bucket_name,object_name,"data.npy")
    except Exception as e:
        print(e)
        return
    data = np.load("data.npy")
    es = run()
    return create_response(status_code=200, content={"content":res})



