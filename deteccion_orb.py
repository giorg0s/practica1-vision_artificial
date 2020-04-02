import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt

CARPETA_TRAIN = 'train'
CARPETA_TEST = 'test'

imagenes_train = []

def carga_imagenes_carpeta(nombre_carpeta):
    print("Se va a iniciar la carga de las imagenes de", nombre_carpeta)
    print("###################################################")
    time.sleep(2)
    for nombre_imagen in os.listdir(nombre_carpeta):
        imagen = cv2.imread(os.path.join(nombre_carpeta, nombre_imagen), 0)
        if imagen is not None:
            imagenes_train.append(imagen)
            print("He leido la imagen ", nombre_imagen)
            time.sleep(.100)
    print("###################################################")
    print("FIN")
    return imagenes_train

def deteccion_orb(imagen):
    orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)
    keypoints = orb.detect(imagen, None)

    keypoints, des = orb.compute(imagen, keypoints)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(imagen, keypoints, None, color=(255, 0, 255), flags= 0)
    plt.imshow(img2), plt.show()


carga_imagenes_carpeta(CARPETA_TRAIN)
deteccion_orb(imagenes_train[15])
print(len(imagenes_train))


