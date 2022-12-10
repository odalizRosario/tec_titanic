import logging
import requests
from fastapi import FastAPI
from fastapi import Body
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s: %(asctime)s|%(name)s|%(message)s")

file_handler = logging.FileHandler("frontend.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)  # Se agrega handler para stream

app = FastAPI()


@app.get("/")
def read_root():
    """
    Funcion de bienvenida que ayuda a recibir las peticiones de raiz
    """
    logger.info("Front-end is all ready to go!")
    return "Front-end is all ready to go!"

@app.get("/v1/healthcheck")
async def v1_healhcheck():
    """
    Funcion que permite verificar el estado de salud del backend
    """
    url3 = settings.ENDPOINT_BACKEND + '/healthcheck'

    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Checking health: {response}")

    return response

def predict_titanic(input):
    url3 = f"{settings.ENDPOINT_BACKEND}/classify_titanic"

    response = requests.post(url3, json=input)
    response = response.text

    return response

@app.post("/v1/classify")
def classify(payload: dict = Body(...)):
    logger.debug(f"Incoming input in the front end: {payload}")
    response = predict_titanic(payload)
    return {"response": response}


