# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
