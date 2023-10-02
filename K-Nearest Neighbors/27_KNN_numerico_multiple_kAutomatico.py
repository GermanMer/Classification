# Importa las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df['price']

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Definir los valores de n_neighbors que deseas probar
param_grid = {'n_neighbors': range(1, 21)}

# Crear un regresor k-NN
knn_regressor = KNeighborsRegressor()

# Realizar una búsqueda en cuadrícula con validación cruzada
grid_search = GridSearchCV(knn_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

# Obtener el mejor valor de n_neighbors encontrado
best_n_neighbors = grid_search.best_params_['n_neighbors']

# Crear un regresor k-NN con el mejor valor de n_neighbors encontrado
best_knn_regressor = KNeighborsRegressor(n_neighbors=best_n_neighbors)

# Entrenar el regresor con tus datos de entrenamiento
best_knn_regressor.fit(x_train, y_train)

# Realizar predicciones en datos desconocidos
y_pred = best_knn_regressor.predict(x_test)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir el mejor valor de n_neighbors encontrado
print(f'Mejor valor de n_neighbors: {best_n_neighbors}')

# Imprimir métricas de evaluación
print(f'Error cuadrático medio (MSE) con el mejor n_neighbors: {mse:.2f}')
print(f'Coeficiente de determinación (R^2) con el mejor n_neighbors: {r2:.2f}')

# Graficar las predicciones junto a los datos reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, y_test, c='blue', label='Datos Reales', marker='o')
plt.scatter(y_test.index, y_pred, c='red', label='Predicciones', marker='x')
plt.xlabel('Índice de Muestra')
plt.ylabel('Precio')
plt.title('Comparación de Datos Reales y Predicciones')
plt.legend()
plt.show()
