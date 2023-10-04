# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Clasificamos las flores en "setosa" (clase 1) y "no setosa" (clase 0)

# Dividir el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar un modelo de regresión logística
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Calcular la curva ROC y el área bajo la curva (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Plotear la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')

# Plotear la distribución de características para la clase positiva y negativa
plt.figure(figsize=(12, 4))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(1, 4, i + 1)
    plt.hist(X[y == 1][:, i], alpha=0.5, label='setosa', color='blue')
    plt.hist(X[y == 0][:, i], alpha=0.5, label='no setosa', color='red')
    plt.title(f'Distribución de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()

# Añadir esta línea para evitar el error de indentación
plt.tight_layout()

# Mostrar la matriz de confusión y el informe de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Informe de clasificación:')
print(class_report)

# Hacer una predicción para un nuevo dato
nuevo_dato = np.array([[5.1, 3.5, 1.4, 0.2]])  # Reemplaza estos valores con tus propios datos
nuevo_dato = scaler.transform(nuevo_dato)  # Escala el nuevo dato como las características originales
prediccion = model.predict(nuevo_dato)
if prediccion[0] == 1:
    print('El nuevo dato es una setosa.')
else:
    print('El nuevo dato no es una setosa.')

# Mostrar los gráficos
plt.show()

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
