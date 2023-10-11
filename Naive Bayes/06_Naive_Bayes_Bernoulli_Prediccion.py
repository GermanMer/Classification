#En este código, hemos agregado una nueva muestra ficticia ("I have a computer science degree") y la hemos vectorizado utilizando el mismo vectorizador que se utilizó para el conjunto de datos original. Luego, hemos utilizado el modelo entrenado (bnb) para predecir la clase de la nueva muestra y hemos impreso la predicción.

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Cargar el conjunto de datos "20 Newsgroups" (seleccione las categorías que desee)
categories = ['sci.med', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Preprocesar los datos de texto utilizando CountVectorizer con características binarias
vectorizer = CountVectorizer(binary=True, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador Naive Bayes Bernoulli
bnb = BernoulliNB()

# Entrenar el clasificador en el conjunto de entrenamiento
bnb.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = bnb.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo Naive Bayes Bernoulli: {accuracy * 100:.2f}%')

# Mostrar el informe de clasificación detallado
class_report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)
print('Informe de clasificación:')
print(class_report)

# Crear una nueva muestra ficticia para hacer una predicción
new_sample = ["I have a computer science degree"]
new_sample_vectorized = vectorizer.transform(new_sample)

# Realizar una predicción para la nueva muestra
new_sample_prediction = bnb.predict(new_sample_vectorized)
print('\nPredicción para la nueva muestra:')
print(f'Clase predicha: {new_sample_prediction[0]} ({newsgroups.target_names[new_sample_prediction[0]]})')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
