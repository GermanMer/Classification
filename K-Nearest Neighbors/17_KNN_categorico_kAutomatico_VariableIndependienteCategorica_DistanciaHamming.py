# Importa las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Codificar la columna 'aspiration' como variables dummy (one-hot)
df = pd.get_dummies(df, columns=['aspiration'], drop_first=True)

# Separar las características x e y
x = df[['aspiration_turbo']]
y = df['fuel-type']

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Definir una lista de valores de n_neighbors que deseas probar
n_neighbors_values = range(1, 21)  # Puedes ajustar este rango según tus necesidades

# Inicializar variables para el mejor valor de n_neighbors y su precisión
best_n_neighbors = None
best_accuracy = 0

# Realizar una búsqueda de n_neighbors
for n_neighbors in n_neighbors_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Calcular la precisión utilizando validación cruzada
    cv_scores = cross_val_score(knn_classifier, x_train, y_train, cv=5)
    mean_cv_accuracy = cv_scores.mean()

    # Actualizar el mejor valor de n_neighbors si encontramos uno mejor
    if mean_cv_accuracy > best_accuracy:
        best_n_neighbors = n_neighbors
        best_accuracy = mean_cv_accuracy

# Crear un clasificador k-NN con el mejor valor de n_neighbors encontrado
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_n_neighbors, metric='hamming')

# Entrenar el clasificador con tus datos de entrenamiento
best_knn_classifier.fit(x_train, y_train)

# Realizar predicciones en datos desconocidos
y_pred = best_knn_classifier.predict(x_test)

# Imprimir el mejor valor de n_neighbors encontrado
print(f'Mejor valor de n_neighbors: {best_n_neighbors}')

# Imprimir la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo con el mejor n_neighbors: {accuracy * 100:.2f}%')

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
