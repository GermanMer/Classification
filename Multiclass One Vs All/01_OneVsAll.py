import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un diccionario para almacenar modelos OvA
ova_models = {}

# Entrenar un modelo OvA para cada clase
for class_label in np.unique(y_train):
    # Crear un clasificador binario para la clase actual vs. el resto
    y_binary_train = (y_train == class_label).astype(int)
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_binary_train)
    ova_models[class_label] = model

# Inicializar una matriz para almacenar las predicciones de cada modelo OvA
num_samples = len(X_test)
num_classes = len(ova_models)
y_pred = np.zeros((num_samples, num_classes))

# Realizar predicciones utilizando los modelos OvA
for idx, (class_label, model) in enumerate(ova_models.items()):
    # Predecir la probabilidad de pertenencia a la clase actual vs. el resto
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred[:, idx] = y_prob

# Determinar la clase final con la probabilidad más alta
final_predictions = np.argmax(y_pred, axis=1)

# Calcular la precisión del modelo OvA
accuracy = accuracy_score(y_test, final_predictions)
print(f'Precisión del modelo OvA: {accuracy * 100:.2f}%')

# Mostrar el informe de clasificación detallado
class_report = classification_report(y_test, final_predictions, target_names=iris.target_names)
print('Informe de clasificación:')
print(class_report)

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
