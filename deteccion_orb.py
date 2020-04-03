import cv2
import numpy as np
import os
import time
import sys
from matplotlib import pyplot as plt

CARPETA_TRAIN = 'train'
CARPETA_TEST = 'test'

# Array para almacenar las imagenes de entrenamiento leidas
imagenes_train = []

# Lo mismo para las imagenes de test
imagenes_test = []

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=12,  # 12
                    key_size=14,  # 20
                    multi_probe_level=2)  # 2

search_params = dict(checks=100)  # or pass empty dictionary


def carga_imagenes_carpeta(nombre_carpeta):
    print("Se va a iniciar la carga de las imagenes de", nombre_carpeta)
    print("###################################################")
    time.sleep(2)
    for nombre_imagen in os.listdir(nombre_carpeta):
        imagen = cv2.imread(os.path.join(nombre_carpeta, nombre_imagen), 0)
        if imagen is not None:
            imagenes_train.append(imagen)
            print("He leido la imagen ", nombre_imagen)
            # time.sleep(.500)
    print("###################################################")
    print("FIN")
    print()
    time.sleep(1)
    
    return imagenes_train


def deteccion_orb(imagenes):
    # se crea el detector ORB
    orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

    keypoints = []

    # Se crea el matcher Flann
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    print("Se va a iniciar la detección ORB")
    print("###################################################")
    time.sleep(2)

    # Se detectan los puntos de interes y se computan los descriptores con ORB
    for i in range(len(imagenes)):
        print("ORB para", i)
        (kp, des) = orb.detectAndCompute(imagenes[i], None)
        keypoints.append(kp)
        flann.add([des])

    # A knnMatch hay que pasarle la estructura con los descriptores almacenados
    matches = flann.knnMatch(np.array([[8, 8, 8]], dtype=np.uint8), k=1)
    
    print("###################################################")
    print("FIN")
    print()
    time.sleep(1)
    
    for m in matches:
        for n in m:
            print("Res - dist:", n.distance, " img: ", n.imgIdx, " queryIdx: ", n.queryIdx, " trainIdx:", n.trainIdx)


def main():
    imgs = carga_imagenes_carpeta(CARPETA_TRAIN)
    deteccion_orb(imgs)


if __name__ == '__main__':
    main()