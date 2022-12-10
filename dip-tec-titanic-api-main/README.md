# Precondiciones
Se asume que ya tiene Python 3+ instalado en su sistema. Si no, por favor, instalelo.  

Revisar este link acorde a su sistema operativo: 
[Python 3 Installation & Setup Guide](https://realpython.com/installing-python/)

Se asume que se tiene instalado Poetry
(https://python-poetry.org/docs/#installation)


# Instalación de ambiente virtual
Abra una terminal, y dirijase a la carpeta raiz del proyecto y ejecute el siguiente comando:

```
poetry env use python
```

# Instalacion de bibliotecas
Para instalar las bibliotecas necesarias, use este comando:
```
poetry install
```

¡Listo, la configuración está lista!

# Ejecutar aplicación

## Frontend

En una terminal ejecutar los siguientes comandos:

```
cd titanic_api/app
```

```
poetry run uvicorn main:app --reload --port 3000

```

Revisar la documentación en la ruta http://127.0.0.1:3000/docs

# Backend

En otra termianl ejecutar los siguientes comandos:

```
cd titanic_api/server
```

```
poetry run uvicorn main:app --reload --port 8000
```

Revisar la documentación en la ruta http://127.0.0.1:8000/docs

# Enviar una solicitud.

(1) Ir a la ruta http://127.0.0.1:3000/docs
(2) Utilizar el endpoint /v1/classify POST
(3) Enviar el siguiente json de prueba

```
{
	"pclass": 1,
	"name": "Sir. Cosmo Edmund (Mr Morgan)",
	"sex": "female",
	"age": 49,
	"sibsp": 1,
	"parch": 0,
	"ticket": "PC 17485",
	"fare": "56.9292",
	"cabin":  "A20",
	"embarked": "C",
	"boat": "1",
	"body": "?",
	"home": "London / Paris"
}
```


## Ejecutar pruebas unitarias y de integración

* Pruebas unitarias y de integración
```
poetry run pytest
```