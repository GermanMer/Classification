#MULTICLASE: la variable dependiente (y) tiene 3 clases, que consta de tres clases diferentes de iris: Setosa, Versicolor y Virginica.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
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

# Creamos un clasificador SVM con kernel lineal
clf = SVC(kernel='linear', probability=True)

# Entrenamos el modelo SVM
clf.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

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
    y_score = clf.decision_function(X_test)
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
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')

# Predicción para un nuevo punto de datos ficticio
nuevo_dato = np.array([[5.5, 3.0]])  # Inventa un nuevo punto de datos
prediccion_nuevo_dato = clf.predict(nuevo_dato)
print("Predicción para el nuevo dato:", prediccion_nuevo_dato)

plt.tight_layout()
plt.show()
