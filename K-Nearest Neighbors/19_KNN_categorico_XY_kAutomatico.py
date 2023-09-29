# Importa las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Definir los valores de n_neighbors que deseas probar
param_grid = {'n_neighbors': range(1, 21)}

# Crear un clasificador k-NN
knn_classifier = KNeighborsClassifier()

# Realizar una búsqueda en cuadrícula con validación cruzada
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Obtener el mejor valor de n_neighbors encontrado
best_n_neighbors = grid_search.best_params_['n_neighbors']

# Crear un clasificador k-NN con el mejor valor de n_neighbors encontrado
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_n_neighbors)

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

# Crear un gráfico de dispersión para visualizar las clasificaciones
plt.figure(figsize=(10, 6))
colors = {'gas': 'blue', 'diesel': 'red'}  # Asignar colores a las categorías
plt.scatter(x_test, y_pred, c=y_test.map(colors), label=y_test, cmap=plt.cm.coolwarm)
plt.xlabel('Compression Ratio')
plt.ylabel('Fuel Type')
plt.title('Clasificación de Fuel Type por Compression Ratio')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Gas', markerfacecolor='blue', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Diesel', markerfacecolor='red', markersize=10)])
plt.show()
