# Importa las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Cargar el conjunto de datos de automóviles desde un archivo CSV
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Codificar las variables categóricas usando one-hot encoding
categorical_columns = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Dividir el DataFrame en características (X) y etiquetas (y)
X = df_encoded.drop(['make_audi', 'make_bmw', 'make_chevrolet',
                     'make_dodge', 'make_honda', 'make_jaguar', 'make_mazda',
                     'make_mercedes-benz', 'make_mitsubishi', 'make_nissan', 'make_peugot',
                     'make_plymouth', 'make_porsche', 'make_saab', 'make_subaru',
                     'make_toyota', 'make_volkswagen', 'make_volvo'], axis=1)
y = df_encoded[['make_audi', 'make_bmw', 'make_chevrolet',
                'make_dodge', 'make_honda', 'make_jaguar', 'make_mazda',
                'make_mercedes-benz', 'make_mitsubishi', 'make_nissan', 'make_peugot',
                'make_plymouth', 'make_porsche', 'make_saab', 'make_subaru',
                'make_toyota', 'make_volkswagen', 'make_volvo']]

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Validación cruzada para evaluar la robustez del modelo
cv_scores = cross_val_score(clf, X, y, cv=5)
print("Puntajes de validación cruzada:", cv_scores)
print("Precisión promedio:", cv_scores.mean())

# Búsqueda de hiperparámetros
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)
print("Mejores hiperparámetros encontrados:", grid_search.best_params_)
best_clf = grid_search.best_estimator_

# Entrenar el modelo con los mejores hiperparámetros
best_clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = best_clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo en prueba:", accuracy)

# Calcular y mostrar el informe de clasificación
classification_rep = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(classification_rep)

# Visualización del árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, rounded=True, class_names=y.columns.tolist(), feature_names=X.columns.tolist())
plt.show()

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
