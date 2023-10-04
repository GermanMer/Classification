# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target  # Clasificamos las flores en tres clases: setosa, versicolor, virginica (0, 1, 2)

# Dividir el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar un modelo de regresión logística multiclase
model = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs', max_iter=10000)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Mostrar la matriz de confusión y el informe de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Informe de clasificación:')
print(class_report)

# Hacer una predicción para un nuevo dato
nuevo_dato = np.array([[5.1, 3.5, 1.4, 0.2]])  # Reemplaza estos valores con tus propios datos
nuevo_dato = scaler.transform(nuevo_dato)  # Escala el nuevo dato como las características originales
prediccion = model.predict(nuevo_dato)
probabilidades = model.predict_proba(nuevo_dato)
clase_predicha = iris.target_names[prediccion][0]

print(f'El nuevo dato es {clase_predicha} con las siguientes probabilidades:')
for clase, probabilidad in zip(iris.target_names, probabilidades[0]):
    print(f'{clase}: {probabilidad:.2f}')

# Plotear la distribución de características para cada clase
plt.figure(figsize=(12, 4))
for i, feature in enumerate(iris.feature_names):
    for target_class in np.unique(y):
        plt.subplot(1, 3, target_class + 1)
        plt.hist(X[y == target_class][:, i], alpha=0.5, label=iris.target_names[target_class])
        plt.title(f'Distribución de {feature} para {iris.target_names[target_class]}')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.legend()

# Añadir esta línea para evitar el error de indentación
plt.tight_layout()

# Calcular las curvas ROC para cada clase
y_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], model.predict_proba(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotear la curva ROC para cada clase
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (AUC = {roc_auc[i]:.2f}) for {iris.target_names[i]}')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para Clasificación Multiclase')
plt.legend(loc='lower right')

# Mostrar los gráficos
plt.show()

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
