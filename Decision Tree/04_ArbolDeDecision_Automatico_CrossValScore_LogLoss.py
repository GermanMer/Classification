# Importa las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
import matplotlib.pyplot as plt

# Carga el conjunto de datos de Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas (clases)

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

# Realiza predicciones en el conjunto de prueba
y_pred = best_clf.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo en prueba:", accuracy)

# Calcula y muestra el informe de clasificación
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Informe de clasificación:")
print(classification_rep)

# Calcula las probabilidades estimadas en lugar de las etiquetas predichas
y_prob = best_clf.predict_proba(X_test)

# Calcula la Log Loss utilizando las probabilidades estimadas y las etiquetas verdaderas
logloss = log_loss(y_test, y_prob)
print("Log Loss:", logloss)

# Visualización del árbol
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
dot_data = export_graphviz(best_clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graphviz.Source(dot_data).view()

# Agrega una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
