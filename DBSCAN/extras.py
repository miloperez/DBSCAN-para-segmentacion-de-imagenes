import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


def getMatDist(X):
    MatDist = np.zeros([len(X), len(X)])

    for i in range(len(X)):
        for j in range(len(X)):
            # MatDist[i][j] = la.norm(X[i] - X[j])
            MatDist[i][j] = math.dist(X[i], X[j])

    return MatDist


def loadMatImg(ImgName):
    img = cv2.imread(ImgName)
    rows, cols, _ = img.shape
    B, G, R = cv2.split(img)
    R = np.reshape(R, (rows * cols, 1))
    G = np.reshape(G, (rows * cols, 1))
    B = np.reshape(B, (rows * cols, 1))
    mat_img = np.column_stack((R, G, B))

    return mat_img


def displayImage(rgbArr):
    plt.imshow(np.reshape(rgbArr, [int(np.sqrt(len(rgbArr))), int(np.sqrt(len(rgbArr))), 3]).astype('uint8'))
    plt.show()

def saveImage(rgbArr, info):
    plt.imshow(np.reshape(rgbArr, [int(np.sqrt(len(rgbArr))), int(np.sqrt(len(rgbArr))), 3]).astype('uint8'))
    plt.savefig(info)

