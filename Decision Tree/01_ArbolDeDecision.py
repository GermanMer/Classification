# Importa las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Carga el conjunto de datos de Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas (clases)

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Entrena el modelo en los datos de entrenamiento
clf.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Calcula y muestra el informe de clasificación
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)
print(classification_rep)

# Puedes visualizar el árbol de decisión si lo deseas
# Esto requiere la instalación de la biblioteca graphviz y pydotplus
# Si no las tienes instaladas, puedes omitir esta parte
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=iris.feature_names,
                           class_names=iris.target_names,
                           filled=True, rounded=True,
                           special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graphviz.Source(dot_data).view()

# Agrega una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
