from pydantic import BaseModel


class Titanic(BaseModel):
    pclass: int
    name: str
    sex: str
    age: int
    sibsp: int 
    parch: int 
    ticket: str
    fare: str
    cabin: str
    embarked: str
    boat: str
    body: str
    home: str