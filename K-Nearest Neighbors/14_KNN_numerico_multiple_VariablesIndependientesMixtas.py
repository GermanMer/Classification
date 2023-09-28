# Importa las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Seleccionar las características 'body-style' y 'horsepower'
x = df[['body-style', 'horsepower']]
y = df['price']

# Codificar la columna 'body-style' como variables dummy (one-hot)
x = pd.get_dummies(x, columns=['body-style'], drop_first=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear un regresor k-NN
knn_regressor = KNeighborsRegressor(n_neighbors=3)  # Aquí, k = 3

# Entrenar el regresor con tus datos de entrenamiento
knn_regressor.fit(x_train, y_train)

# Realizar predicciones en datos desconocidos
y_pred = knn_regressor.predict(x_test)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir métricas de evaluación
print(f'Error cuadrático medio (MSE): {mse:.2f}')
print(f'Coeficiente de determinación (R^2): {r2:.2f}')

# Graficar las predicciones junto a los datos reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, y_test, c='blue', label='Datos Reales', marker='o')
plt.scatter(y_test.index, y_pred, c='red', label='Predicciones', marker='x')
plt.xlabel('Índice de Muestra')
plt.ylabel('Precio')
plt.title('Comparación de Datos Reales y Predicciones')
plt.legend()
plt.show()
