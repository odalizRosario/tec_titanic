import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parents[3]))
sys.path.append(str(pathlib.Path(__file__).parents[1]))

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder as oneHot
from config import settings
from models.models import Titanic


# Creando transformadores.

class ChangeDataType(BaseEstimator, TransformerMixin):

    """
    Transformer que permite modificar el tipo de dato a las columnas indicadas
    Keyword arguments:
    variable: Diccionario de (k,v) = (column,datatype)
    Return: Nuevo dataframe transformado.
    """

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        for var in self.variables:
                X[var] = X[var].astype('float')
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):

    """
    Transformer que permite agregar la palabra missing en caso de un valor nulo
    Keyword arguments:
    variable: Lista de columnas a modificar
    Return: Nuevo dataframe transformado.
    """

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X


class ReplaceNAN(BaseEstimator, TransformerMixin):

    """
    Transformer que permite remplazar el caracter '?' por nan
    Keyword arguments:
    variable: Lista de columnas a modificar
    Return: Nuevo dataframe transformado.
    """

    def __init__(self, variables=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var] = X[var].replace("?", np.nan)
        return X


class CategoricalTransformer(BaseEstimator, TransformerMixin):

    """
    Transformer que permite obtener el titulo del nombre de una persona o traducir la abreviación de embarcadero.
    Keyword arguments:
    variable_fit: Nombre de la columna a utilizar para input de la transformación
    variable_transform: Nombre de la columna asignada al resultado de la transformación
    Return: Nuevo dataframe transformado.
    """

    def __init__(self, variable_fit, variable_transform):
        self.variable_fit = variable_fit
        self.variable_transform = variable_transform

    def get_transform(self, line, mode):

        if self.variable_fit == 'name':
            if re.search('Mrs', line):
                return 'Mrs'
            elif re.search('Mr', line):
                return 'Mr'
            elif re.search('Miss', line):
                return 'Miss'
            elif re.search('Master', line):
                return 'Master'

        elif self.variable_fit == 'embarked':
            if re.search('C', line):
                return 'Cherbourg'
            elif re.search('Q', line):
                return 'Queenstown'
            elif re.search('S', line):
                return 'Southampton'

        return self.get_transform(mode, None)

    def fit(self, X, Y=None):
        mode = X[self.variable_fit].mode()[0]
        self.new_name = [self.get_transform(
            name, mode) for name in X[self.variable_fit]]
        return self

    def transform(self, X):
        X[self.variable_transform] = pd.Series(self.new_name)
        return X


class NumericalImputer(BaseEstimator, TransformerMixin):

    """
    Transformer que permite inputar la media a columnas numericas cuado se encuenta un valor nan
    Keyword arguments:
    variables: Nombres de las columnas a transformar
    Return: Nuevo dataframe transformado.
    """

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.median_dict_ = {}
        for var in self.variables:
            self.median_dict_[var] = X[var].median()
        return self

    def transform(self, X):
        for var in self.variables:
            X[var] = X[var].fillna(self.median_dict_[var])
        return X


class DropColumns(BaseEstimator, TransformerMixin):

    """
    Transformer que permite eliminar columnas
    Keyword arguments:
    variables: Nombres de las columnas a eliminar.
    Return: Nuevo dataframe transformado.
    """

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.columns,self.variables)
        X.drop(self.variables, axis=1, inplace=True)
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):

    """
    Transformer que permite aplicar la tecnica One hot encoder a una lista de columnas.
    Keyword arguments:
    variables: Nombres de las columnas a transformar.
    Return: Nuevo dataframe transformado.
    """
    def __init__(self, variables):
        self.encoder = oneHot(handle_unknown="ignore", drop="first")
        self.variables = variables
        self.feature_names_out = []

    def fit(self, X: pd.DataFrame, y=0) -> None:
        self.feature_names_out = X.columns
        self.encoder.fit(X[self.variables])
        self.variables_out = self.encoder.get_feature_names_out(self.variables)
        return self

    def transform(self, X: pd.DataFrame, y=0) -> None:
        X[self.encoder.get_feature_names_out(self.variables)] = self.encoder.transform(
            X[self.variables]
        ).toarray()
        X.drop(self.variables, axis=1, inplace=True)
        return X

    def get_feature_names_out(self, x):
        return self.feature_names_out


class titanicClassifier():

    def __init__(self):
        data = pd.read_csv(settings.URL)
        
        #Problema en model, en declarar variable con '.'
        data.rename(columns = {'home.dest':'home'}, inplace = True)

        X_train, X_test, y_train, y_test = train_test_split(
                                                     data.drop(settings.TARGET, axis=1),
                                                     data[settings.TARGET],
                                                     test_size=settings.SPLIT_TEST
                                                    )
        self.X, self.y = X_train.reset_index(), y_train.reset_index()
        self.clf = self.train_model()
        self.titanic_type = {
            0: 'not survived',
            1: 'survived'
            }

    def train_model(self) -> RandomForestClassifier:
        """Funcion en cargada de entrenar el modelo
        Keyword arguments:
        argument -- description
        Return: Classifier titanic in random forest
        """
        titanic_pipeline = Pipeline(
                              [
                                ('Replace nan', ReplaceNAN(settings.REPLACE_NAN)),
                                ('Transform_name_title', CategoricalTransformer('name','title')),
                                ('Imputer_embarked', CategoricalImputer('embarked')),
                                ('Categorical_imputer', CategoricalTransformer('embarked','embarked')),
                                ('Numerical imputer', NumericalImputer(settings.FILEDS_NUMBER)),
                                ('Change data type', ChangeDataType(settings.FILEDS_NUMBER)),
                                ('One hot encoder', OneHotEncoder(settings.ONE_HOT_ENCODER)),
                                ('Ddrop columns', DropColumns(settings.DROP_COLS)),
                                ('Random forest', RandomForestClassifier(max_depth=2, random_state=0))
                              ])
        return titanic_pipeline.fit(self.X, self.y)

    
    def classify_titanic(self, titanic: Titanic):
        """
        Funcion encargada de recibir los datos de un pasajero del titanic e indicar si sobrevirá o no.
        """
        X = {   'index': 0,
                'home':  titanic.home,
                'pclass':  titanic.pclass,
                'name':  titanic.name,
                'sex':  titanic.sex,
                'age':  titanic.age,
                'sibsp':  titanic.sibsp,
                'parch':  titanic.parch,
                'ticket':  titanic.ticket,
                'fare':  titanic.fare,
                'cabin':  titanic.cabin,
                'embarked':  titanic.embarked,
                'boat':  titanic.boat,
                'body':  titanic.body
            }
        prediction = self.clf.predict(pd.DataFrame([X]))

        print(f'class: {self.titanic_type[prediction[0][1]]}')
        return {'class': self.titanic_type[prediction[0][1]]}