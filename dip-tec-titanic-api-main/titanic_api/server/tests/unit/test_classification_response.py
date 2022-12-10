import pytest
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from classifier.titanic_classifier import titanicClassifier
from models.models import Titanic


def get_test_response_data() -> list:
    return [('survived',1,'Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")','male',49,1,0,'PC 17485','56.9292','A20','C','1','?','London / Paris'),
            ('not survived',2,'McKane, Mr. Peter David','male',46,0,0,'28403','26','?','S','?','?','Rochester, NY')]

@pytest.mark.parametrize(
    "survived,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,home",
    get_test_response_data(),
)
def test_response_parametrize(
    survived: any,
    pclass: any,
    name: any,
    sex: any,
    age: any,
    sibsp: any,
    parch: any,
    ticket: any,
    fare: any,
    cabin: any,
    embarked: any,
    boat: any,
    body: any,
    home: any
) -> None:
    titanic = Titanic(
        pclass = pclass,
        name = name,
        sex = sex,
        age = age,
        sibsp = sibsp,
        parch = parch,
        ticket = ticket,
        fare = fare,
        cabin = cabin,
        embarked = embarked,
        boat = boat,
        body = body,
        home = home,
    )

    classifier = titanicClassifier()

    assert type(classifier.classify_titanic(titanic)["class"]) == type(survived)
