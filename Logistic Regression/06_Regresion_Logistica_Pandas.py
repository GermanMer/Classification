import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Cargar el DataFrame desde el archivo CSV y eliminar filas con valores nulos
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Filtrar el DataFrame para mantener solo las filas con 'fuel-type' igual a 'gas' o 'diesel'
df = df[df['fuel-type'].isin(['gas', 'diesel'])]

# Mapear 'fuel-type' a una variable binaria (0 para 'gas' y 1 para 'diesel')
df['fuel-type'] = df['fuel-type'].map({'gas': 0, 'diesel': 1})

# Seleccionar columnas categóricas y numéricas
categorical_cols = ['make', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'aspiration', 'engine-type', 'num-of-cylinders', 'fuel-system']
numeric_cols = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# Definir transformaciones para columnas categóricas y numéricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Aplicar transformaciones a las columnas correspondientes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Crear un modelo de regresión logística binaria
model = LogisticRegression(random_state=42)

# Crear el pipeline completo
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# Dividir el DataFrame en características (X) y etiquetas (y)
X = df.drop('fuel-type', axis=1)  # Características
y = df['fuel-type']  # Variable dependiente ('fuel-type')

# Dividir el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Mostrar la matriz de confusión y el informe de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Informe de clasificación:')
print(class_report)

# Calcular las probabilidades de predicción para el conjunto de prueba
y_proba = clf.predict_proba(X_test)[:, 1]

# Calcular el AUC (Área bajo la curva ROC)
roc_auc = roc_auc_score(y_test, y_proba)
print(f'AUC: {roc_auc:.2f}')

# Calcular la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)

# Plotear la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para Clasificación Binaria')
plt.legend(loc='lower right')

# Plotear la distribución de características para la clase positiva y negativa
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numeric_cols):
    plt.subplot(4, 4, i + 1)
    plt.hist(X_test[y_test == 1][feature], color='b', alpha=0.5, label='diesel', bins=20)
    plt.hist(X_test[y_test == 0][feature], color='r', alpha=0.5, label='gas', bins=20)
    plt.title(f'Distribución de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()

# Añadir esta línea para evitar el error de indentación
plt.tight_layout()

# Muestra el gráfico
plt.show()

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
