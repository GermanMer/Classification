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
X = iris.data
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
plt.figure(figsize=(10, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {iris.target_names[i]}) (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
