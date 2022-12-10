import logging

from fastapi import FastAPI
from starlette.responses import JSONResponse

from classifier.titanic_classifier import titanicClassifier as TitanicClassifier
from models.models import Titanic

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s: %(asctime)s|%(name)s|%(message)s")

file_handler = logging.FileHandler("server.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)  # Se agrega handler para stream

app = FastAPI()

titanicModel = TitanicClassifier()


@app.get("/")
def read_root():
    """
    Funcion de bienvenida que ayuda a recibir las peticiones de raiz
    """
    return "Iris classifier is all ready to go!"


@app.get("/healthcheck", status_code=200)
async def healthcheck():
    """
    Funcion que permite verificar el estado de salud del backend
    """
    logger.info("Servers is all ready to go!")
    return "Iris classifier is all ready to go!"


@app.post("/classify_titanic")
async def classify(titanic_features: Titanic):
    logger.debug(f"Incoming titanic features to the server: {titanic_features}")
    response = JSONResponse(titanicModel.classify_titanic(titanic_features))
    logger.debug(f"Outgoing classification from the server: {response}")
    return response
