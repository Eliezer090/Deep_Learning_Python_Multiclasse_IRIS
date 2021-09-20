import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json

#importa a base
base = pd.read_csv('/Users/es19237/Desktop/Deep Learning/Classificacao Mais Classes/Files/iris.csv')
#Separa os valores do que é esses valores
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

#Criando a rede, com os melhores parametros do tuning
classificador = Sequential()
classificador.add(Dense(units = 10, activation = 'relu', kernel_initializer='random_uniform',input_dim = 4))
classificador.add(Dense(units = 10, activation = 'relu', kernel_initializer='random_uniform'))
#Criando a saida, aqui teremos 3 pois temos 3 tipos de plantas.
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

classificador.fit(previsores,classe_dummy, batch_size = 10,epochs = 1000)
#Saida de loss: 0.04 e accuracy: 0.98

#Salvar para classificar
classificador_json = classificador.to_json()
with open("classificador_iris.json", "w") as json_file:
    json_file.write(classificador_json)
classificador.save_weights("classificador_iris.h5")


### Carrega os arquivos para classificar ####
arquivo = open('classificador_iris.json', 'r')
estrutura_classificador = arquivo.read()
#para limpar memoria
arquivo.close()
classificador_carregado = model_from_json(estrutura_classificador)
classificador_carregado.load_weights("classificador_iris.h5")

#Classificar novo registro com base na rede salva
novo = np.array([[1.2, 3.5, 8.9, 1.1]])
previsao = classificador_carregado.predict(novo)

previsao = (previsao > 0.5)
if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')