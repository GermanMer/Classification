import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets

# Cargar el conjunto de datos Iris como un DataFrame de pandas
iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Dividir el DataFrame en características (X) y etiquetas (y)
X = df.drop(columns='target')
y = df['target']

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

# Crear una nueva muestra ficticia como un DataFrame de pandas
new_sample = pd.DataFrame([[5.0, 3.0, 4.0, 1.0]], columns=iris['feature_names'])

# Realizar una predicción para la nueva muestra
new_sample_predictions = ovo_classifier.decision_function(new_sample)

# Obtener la clase predicha basada en la decisión más cercana a cero
predicted_class = np.argmax(new_sample_predictions)

# Obtener las decisiones de los clasificadores binarios
decision_values = ovo_classifier.decision_function(X_test)

# Contar los votos de los clasificadores para la nueva muestra
votes = np.zeros(len(iris.target_names), dtype=int)
for i in range(len(iris.target_names)):
    for j in range(i + 1, len(iris.target_names)):
        decision_i = decision_values[:, i]
        decision_j = decision_values[:, j]

        # Verificar si el clasificador i vota por la clase i
        votes[i] += np.sum(decision_i > decision_j)

        # Verificar si el clasificador j vota por la clase j
        votes[j] += np.sum(decision_j > decision_i)

# Imprimir la clase predicha, la decisión de los clasificadores y los votos para la nueva muestra
print('\nPredicción para la nueva muestra:')
print(f'Clase predicha: {predicted_class} ({iris.target_names[predicted_class]})')
print('Decisiones de los clasificadores:')
for idx, (class_name, decision_value) in enumerate(zip(iris.target_names, decision_values[0])):
    print(f'{class_name}: {decision_value:.4f}')
print('Votos de los clasificadores:')
for idx, (class_name, vote_count) in enumerate(zip(iris.target_names, votes)):
    print(f'{class_name}: {vote_count} votos')

# Crear una gráfica de barras para visualizar las probabilidades en la nueva muestra
plt.figure(figsize=(8, 5))
plt.bar(iris.target_names, new_sample_predictions[0], color='skyblue')
plt.title('Probabilidades para la Nueva Muestra')
plt.xlabel('Clases')
plt.ylabel('Probabilidad')
plt.show()
