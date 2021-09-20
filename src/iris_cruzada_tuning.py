import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('./Files/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer=kernel_initializer,input_dim = 4))
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer=kernel_initializer))
    #saida
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, loss = loss,
                          metrics = ['categorical_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 30], 'epochs': [100, 1000],
              'optimizer': ['adam'],
              'loss': ['categorical_crossentropy'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh','softmax'],
              'neurons': [5, 10,15]
              }

grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros, scoring='accuracy', cv=10)

grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_

melhor_precisao = grid_search.best_score_
