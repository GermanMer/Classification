import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador multiclase OvO utilizando la regresión logística como clasificador binario base
ovo_classifier = OneVsOneClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))

# Entrenar el clasificador OvO en el conjunto de entrenamiento
ovo_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = ovo_classifier.predict(X_test)

# Calcular la precisión del modelo OvO
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo OvO: {accuracy * 100:.2f}%')

# Mostrar el informe de clasificación detallado
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Informe de clasificación:')
print(class_report)

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
