#En este ejemplo, hemos creado un clasificador Naive Bayes Bernoulli y hemos especificado el hiperparámetro binarize con un valor de umbral de 2.5. Esto significa que cualquier característica en los datos que tenga un valor mayor que 2.5 se considerará como 1, mientras que las características con valores iguales o menores que 2.5 se considerarán como 0.
#La elección del valor de umbral 2.5 es arbitraria y puede ajustarse según los requisitos de tu problema y la naturaleza de los datos. El ajuste de este valor puede influir en el rendimiento del modelo en función de cómo deseas que se binaricen las características.

#Hemos creado una nueva muestra ficticia con características numéricas y luego hemos utilizado el modelo entrenado (bnb) para predecir la clase de la nueva muestra. El resultado de la predicción para la nueva muestra se imprime al final del código.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador Naive Bayes Bernoulli con el parámetro binarize
binarize_threshold = 2.5  # Umbral para binarizar las características
bnb = BernoulliNB(binarize=binarize_threshold)

# Entrenar el clasificador en el conjunto de entrenamiento
bnb.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = bnb.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo Naive Bayes Bernoulli: {accuracy * 100:.2f}%')

# Mostrar el informe de clasificación detallado
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Informe de clasificación:')
print(class_report)

# Crear una nueva muestra ficticia para hacer una predicción
new_sample = np.array([[2.8, 3.0, 1.0, 0.2]])  # Aquí debes proporcionar las características de la nueva muestra

# Realizar una predicción para la nueva muestra
new_sample_prediction = bnb.predict(new_sample)
print('\nPredicción para la nueva muestra:')
print(f'Clase predicha: {new_sample_prediction[0]} ({iris.target_names[new_sample_prediction[0]]})')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
