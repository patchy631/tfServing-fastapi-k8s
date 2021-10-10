import uvicorn
from fastapi import FastAPI

from services import prediction

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, This is a DTFR classification API'}


@app.get('/predict/featureId/{featureId}')
def predict_dtfr_for_featureId(featureId):
    return prediction.get_dtfr_for_featureId(featureId)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
