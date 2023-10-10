#En este código, hemos creado una nueva muestra ficticia con características numéricas y luego hemos utilizado el modelo entrenado (gnb) para predecir la clase de la nueva muestra. Hemos impreso el resultado de la predicción para la nueva muestra al final del código.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador Naive Bayes Gaussiano
gnb = GaussianNB()

# Entrenar el clasificador en el conjunto de entrenamiento
gnb.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = gnb.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo Naive Bayes: {accuracy * 100:.2f}%')

# Mostrar el informe de clasificación detallado
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Informe de clasificación:')
print(class_report)

# Crear una nueva muestra ficticia para hacer una predicción
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Aquí debes proporcionar las características de la nueva muestra

# Realizar una predicción para la nueva muestra
new_sample_prediction = gnb.predict(new_sample)
print('\nPredicción para la nueva muestra:')
print(f'Clase predicha: {new_sample_prediction[0]} ({iris.target_names[new_sample_prediction[0]]})')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
