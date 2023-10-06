#MULTICLASE: la variable dependiente (y) tiene 3 clases, que consta de tres clases diferentes de iris: Setosa, Versicolor y Virginica.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mpl_toolkits.mplot3d import Axes3D

# Cargamos el conjunto de datos Iris
iris = load_iris()
X = iris.data[:, :3]  # Tomamos las tres primeras características
y = iris.target

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizamos las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creamos un clasificador SVM con kernel lineal
clf = SVC(kernel='linear', probability=True)

# Entrenamos el modelo SVM
clf.fit(X_train, y_train)

# Preparamos una malla 3D para el gráfico de la frontera de decisión
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
z_min, z_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), np.arange(z_min, z_max, 0.1))

# Predecir la clase para cada punto en la malla 3D
Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Calcular la exactitud en el conjunto de prueba
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Generar un informe de clasificación
class_report = classification_report(y_test, y_pred)

# Crear una figura 3D
fig = plt.figure(figsize=(12, 6))

# Subplot para el gráfico de dispersión tridimensional
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap=plt.cm.coolwarm)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Scatter Plot with Decision Boundary (3D)')

# Dibujar los vectores de soporte
support_vectors = clf.support_vectors_
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], support_vectors[:, 2], c='k', marker='x', s=100)

# Subplot para mostrar las métricas
ax2 = fig.add_subplot(122)
ax2.text(0.5, 0.9, f'Accuracy: {accuracy:.2f}', fontsize=12, ha='center', va='center')
ax2.text(0.5, 0.7, 'Confusion Matrix:', fontsize=12, ha='center', va='center')
ax2.text(0.5, 0.6, str(conf_matrix), fontsize=12, ha='center', va='center')
ax2.text(0.5, 0.4, 'Classification Report:', fontsize=12, ha='center', va='center')
ax2.text(0.1, 0.1, class_report, fontsize=10)

plt.tight_layout()
plt.show()
