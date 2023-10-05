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
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
