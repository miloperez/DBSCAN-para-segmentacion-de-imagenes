import numpy as np
import DBSCAN as db
import extras
from matplotlib import pyplot as plt

# para cargar cualquier archivo en la carpeta
archivo = "checker100.png"
MatImg = extras.loadMatImg(archivo)


# imagen original por comparación
plt.title("Original")
extras.displayImage(MatImg)


# hiperparámetros
eps = 6.75
minPts = 15

# print(f'eps = {eps}, minPts = {minPts}')
prueba = db.DBSCAN(len(MatImg), MatImg, eps, minPts)

prueba.getMatDist()

# DBSCAN como tal
k = prueba.getVecTags()

# a partir de aquí construye una imagen con base en el vector de etiquetas

color = None

aux = np.zeros([len(k), 3])

for i in np.unique(k):
    for j in range(len(k)):
        if k[j] == i:
            if color is None:
                color = MatImg[j]
            aux[j] = color
    color = None

plt.title(f"Procesada eps = {eps}, minPts = {minPts} grupos {len(np.unique(k))}")
extras.displayImage(aux)
