import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Cargamos el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Utilizamos solo las dos primeras características
y = iris.target

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

# Calculamos la curva ROC y el área bajo la curva (AUC) para cada clase
n_classes = len(iris.target_names)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    y_score = best_svm.decision_function(X_test)
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Dibujar la curva ROC y mostrar el AUC
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(n_classes):
    label = 'ROC curve (class %s) (AUC = %0.2f)' % (iris.target_names[i], roc_auc[i])
    plt.plot(fpr[i], tpr[i], label=label)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Gráfico de Frontera de Decisión (Decision Boundary)
plt.subplot(1, 2, 2)
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')

plt.tight_layout()
plt.show()
