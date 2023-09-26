# Importa las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['compression-ratio']]
y = df['fuel-type']

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear un clasificador k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Aquí, k = 3

# Entrenar el clasificador con tus datos de entrenamiento
knn_classifier.fit(x_train, y_train)

# Realizar predicciones en datos desconocidos
y_pred = knn_classifier.predict(x_test)

# Imprimir la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Imprimir métricas de evaluación
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Imprimir el reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Graficar las predicciones junto a los datos reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, y_test, c='blue', label='Datos Reales', marker='o')
plt.scatter(y_test.index, y_pred, c='red', label='Predicciones', marker='x')
plt.xlabel('Índice de Muestra')
plt.ylabel('Tipo de Combustible')
plt.title('Comparación de Datos Reales y Predicciones')
plt.legend()
plt.show()
