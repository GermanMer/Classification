import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Cargamos el conjunto de datos de cáncer de mama de Wisconsin
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Selecciona solo las dos primeras características
X = X[:, :2]

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizamos las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definimos la cuadrícula de hiperparámetros a explorar
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']}

# Creamos un clasificador SVM
svm = SVC(probability=True)

# Creamos un objeto GridSearchCV para la búsqueda de cuadrícula
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Entrenamos el modelo SVM con búsqueda de cuadrícula
grid_search.fit(X_train, y_train)

# Obtenemos el mejor modelo con los hiperparámetros óptimos
best_svm = grid_search.best_estimator_

print("Parámetros óptimos:", grid_search.best_params_)

# Realizamos predicciones en el conjunto de prueba
y_pred = best_svm.predict(X_test)

# Calculamos la exactitud (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generamos el informe de clasificación
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Calculamos la curva ROC y el área bajo la curva (AUC)
fpr, tpr, _ = roc_curve(y_test, best_svm.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Dibujar la curva ROC y mostrar el AUC
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Dibujar la frontera de decisión
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')

# Dibujar los vectores de soporte
plt.scatter(best_svm.support_vectors_[:, 0], best_svm.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Support Vectors')

# Crear un nuevo punto de datos (debes ajustar los valores)
new_data_point = np.array([[5.5, 3.0]])

# Estandarizar el nuevo punto de datos usando el mismo scaler
new_data_point = scaler.transform(new_data_point)

# Hacer una predicción para el nuevo punto de datos
new_prediction = best_svm.predict(new_data_point)

# Imprimir el resultado de la predicción
print("Predicción para el nuevo punto de datos:", new_prediction)

plt.tight_layout()
plt.show()
