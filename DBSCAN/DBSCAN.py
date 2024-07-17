import numpy as np
import collections as coll
import math


class DBSCAN:
    # inicializador requiere un arreglo de datos, eps para el radio
    # y la cantidad mínima de datos que conforman un vecindario
    def __init__(self, n, arr, eps, MinPts):
        self.n = n
        self.Mat = arr
        self.eps = eps
        self.MinPts = MinPts
        # inicializar arreglo de tags con el valor -1 para indicar que no se han revisado
        self.nClusters = int(0)
        self.tags = np.full([n], -1)

    # string function regresa información de la clase
    def __str__(self):
        return f"DBSCAN with {self.eps} radius and {self.MinPts} MinPts.\n {len(self.Mat)} elements loaded"

    # a partir del arreglo de datos obtiene la matriz de distancias
    # se mantiene como un método por si se quiere inicializar la clase con la matriz de distancias directamente
    def getMatDist(self):
        MatDist = np.zeros([len(self.Mat), len(self.Mat)])

        for i in range(len(self.Mat)):
            for j in range(len(self.Mat)):
                MatDist[i][j] = math.dist(self.Mat[i], self.Mat[j])

        self.Mat = MatDist
        return self.Mat

    # obtener lista con iteradores de los vecinos del elemento indicado
    # requiere un vec de la matDist
    def getNB(self, vec):
        list = coll.deque()
        for i in range(len(vec)):
            if vec[i] <= self.eps and vec[i] != 0:
                list.append(i)

        return np.array(list)

    # regresa el número de vecinos en el vecindario eps y distancia distinta de cero
    # requiere un vec de la matDist
    def countNB(self, vec):
        auxvec = vec[np.where(vec <= self.eps)]
        return len(auxvec[np.where(auxvec > 0)])

    # regresar lista de índices de puntos centrales
    def getCorePoints(self):
        list = coll.deque()
        for i in range(self.n):
            if self.countNB(self.Mat[i]) >= self.eps:
                list.append(i)
        return np.array(list)

    # de la lista de índices de puntos centrales, regresa los que no se han visitado
    def getNotVisitedCores(self):
        list = coll.deque()
        cp = self.getCorePoints()
        for i in cp:
            if self.tags[i] == -1:
                list.append(i)

        return np.array(list)

    # calcula los clusters, regresa el vector de etiquetas
    def getVecTags(self):
        cola = coll.deque()
        nvc = self.getNotVisitedCores()

        while len(nvc) != 0:
            # print(f'len(nvc){len(nvc)}')
            # print(f'tags {self.tags}')
            # print("nuevo x")
            x = np.random.choice(nvc)
            # asignar nuevo cluster
            self.tags[x] = self.nClusters + 1
            self.nClusters += 1

            # agregar a la cola los vecinos de x
            for j in self.getNB(self.Mat[x]):
                cola.append(j)

            # print(f'cola {cola}')
            # mientras la cola tenga elementos
            while len(cola) != 0:
                # print(f'len(cola){len(cola)}')
                # se les asigna la etiqueta de x al elemento de la cola
                self.tags[cola[0]] = self.tags[x]

                # se añaden a la cola los vecinos del elemento analizado
                # print(f'cola[0] {cola[0]}')
                for a in self.getNB(self.Mat[cola[0]]):
                    if a not in cola and self.tags[a] == -1:
                        cola.append(a)
                # se sale de la cola el elemento ya que se analizaron a todos sus vecinos
                cola.popleft()

            nvc = self.getNotVisitedCores()

        # print(self.tags)
        # print(len(np.unique(self.tags)))
        return self.tags
