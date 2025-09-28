import gzip
import pickle

try:
    # Intentar cargar el modelo
    with gzip.open("files/models/model.pkl.gz", "rb") as file:
        model = pickle.load(file)
    print("El archivo se deserializó correctamente. El modelo es válido.")
except Exception as e:
    print(f"Error al deserializar el archivo: {e}")

import sys
import sklearn

print(f"Python version: {sys.version}")
print(f"scikit-learn version: {sklearn.__version__}")