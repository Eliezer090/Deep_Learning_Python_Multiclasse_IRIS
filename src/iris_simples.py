import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix

base = pd.read_csv('/Users/es19237/Desktop/Deep Learning/Classificacao Mais Classes/Files/iris.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe_dummy,test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units=4, activation='relu', input_dim = 4))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax'))

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento,batch_size=10,epochs=1000)

resultado = classificador.evaluate(previsores_teste,classe_teste)

previsores = classificador.predict(previsores_teste)

previsores = (previsores>0.5)

classe_teste2 = [np.argmax(t) for t in classe_teste]
previsores2 = [np.argmax(t) for t in previsores]
matriz = confusion_matrix(classe_teste2,previsores2)
