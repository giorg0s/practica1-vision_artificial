import cv2
import numpy as np
import os
import time
import sys
from matplotlib import pyplot as plt

CARPETA_TRAIN = 'train'
CARPETA_TEST = 'test'

# Parámetros FLANN
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=12,  # 12
                    key_size=15,  # 20
                    multi_probe_level=2)  # 2

search_params = dict(checks=100)  # or pass empty dictionary

# Almacenamiento de puntos de interes
training_keypoints = []

# Estructura para almacenar los descriptores (basada en Flann)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def carga_imagenes_carpeta(nombre_carpeta):
    imagenes = []

    print("Se va a iniciar la carga de las imagenes de", nombre_carpeta)
    print("###################################################")
    time.sleep(2)

    for nombre_imagen in os.listdir(nombre_carpeta):
        imagen = cv2.imread(os.path.join(nombre_carpeta, nombre_imagen), 0)
        if imagen is not None:
            imagenes.append(imagen)
            print("He leido la imagen ", nombre_imagen)
            # time.sleep(.500)
    print("###################################################")
    print("FIN")
    print()
    time.sleep(1)

    return imagenes


def entrenamiento_orb(training_imgs):
    # se crea el detector ORB
    orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

    print("Se va a iniciar la detección ORB")
    print("###################################################")
    #time.sleep(2)

    # Se detectan los puntos de interes y se computan los descriptores con ORB
    for i, img in enumerate(training_imgs):
        print("ORB para", i)
        (kp, des) = orb.detectAndCompute(img, None)
        training_keypoints.append(kp)  # se guarda la informacion de cada keypoint
        flann.add([des])  # se almacenan los descriptores

    print("###################################################")
    print("FIN")
    print()
    #time.sleep(1)


def procesamiento_img(test_imgs):
    orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)
    vector_votacion = []

    # Ha knnMatch hay que pasarle la estructura con los descriptores almacenados
    for i, img in enumerate(test_imgs):
        print("Para la imagen de test", i)

        (kp, des) = orb.detectAndCompute(img, None)

        matches = flann.knnMatch(des, k=2)


        # HAY QUE DEFINIR LA VOTACION DE HOUGH PARA QUEDARSE CON EL MEJOR DESCRIPTOR

        #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
        #plt.imshow(img3)

        # Se compara el descriptor de la imagen leida con todos los que contiene la estructura Flann (los coches que ya conoce)
        #for m in matches:
            #for n in m:
                #print("Res - dist:", n.distance, " img: ", n.imgIdx, " queryIdx: ", n.queryIdx, " trainIdx:", n.trainIdx)


def main():
    training_imgs = carga_imagenes_carpeta(CARPETA_TRAIN)
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST)

    entrenamiento_orb(training_imgs)
    procesamiento_img(test_imgs)


if __name__ == '__main__':
    main()
