# /usr/bin/python3

import cv2
import numpy as np
import os
import time
import statistics as stats
import math

CARPETA_TRAIN = 'img/train'
CARPETA_TEST = 'img/test'

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
    training_img_size_x = []
    training_img_size_y = []

    # Se crea el detector ORB
    orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

    print("Se va a iniciar la detección ORB")
    print("###################################################")

    for i, img in enumerate(training_imgs):
        training_img_size_x.append(img.shape[1])
        training_img_size_y.append(img.shape[0])

        # Se detectan los puntos de interes y se computan los descriptores con ORB
        print("ORB para", i)
        (kps_training, des_training) = orb.detectAndCompute(img, None)
        training_keypoints.append(kps_training)  # se guarda la informacion de cada keypoint
        flann.add([des_training])  # se almacenan los descriptores

    valor_x = list(set(training_img_size_x))[0] # Este valor se corresponde con la anchura de la imagen de training
    valor_y = list(set(training_img_size_y))[0] # Este valor se corresponde con la altura de la imagen de training

    print("###################################################")
    print("FIN")
    print()

    return (valor_x, valor_y) # se devuelve el tamano de la imagen de entrenamiento (en este caso son todas iguales)


def votacion_hough(centro, training_kps, kact):
    vector = (centro[0] - training_kps.pt[0], centro[1] - training_kps.pt[1])
    vector = (vector[0] * kact.size / training_kps.size, vector[1] * kact.size / training_kps.size)

    angulo = np.rad2deg(math.atan2(vector[1], vector[0])) + (kact.angle - training_kps.angle)

    modulo = np.sqrt(vector[0] * vector[0] + vector[1] * vector[1])

    vector = (modulo * np.cos(angulo) + kact.pt[0], modulo * np.sin(angulo) + kact.pt[1])
    vector = (np.uint8(vector[0] / 10), np.uint8(vector[1] / 10))

    return vector


def procesamiento_img(test_imgs, training_x, training_y):
    orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

    # Ha knnMatch hay que pasarle la estructura con los descriptores almacenados
    for i, img in enumerate(test_imgs):
        print("=============================================================================")
        print("Para la imagen de test", i)

        img = cv2.resize(img, dsize=(training_x, training_y), interpolation=cv2.INTER_CUBIC)

        (kps_test, des_test) = orb.detectAndCompute(img, None)
        par = zip(kps_test, des_test)

        img_size_y = np.uint8(test_imgs[i].shape[0] / 10)
        img_size_x = np.uint8(test_imgs[i].shape[1] / 10)

        # Se crea el vector de votacion a partir del tamano de la imagen reducido por un factor (en este caso 10 que es
        # el que determina el enunciado)
        vector_votacion = np.zeros((img_size_y, img_size_x), dtype=np.uint8)

        for (kp, des) in par:
            # Se obtienen los matches potenciales para cada imagen de test
            matches = flann.knnMatch(des, k=5)

            for vecinos in matches:
                for m in vecinos:
                    vector = votacion_hough((225,110), training_keypoints[m.imgIdx][m.trainIdx], kp)
                    if (vector[0] >= 0) & (vector[1] >= 0) & (vector[0] < (img_size_y - 1)) & (
                            vector[1] < (img_size_x - 1)):
                        vector_votacion[vector[0]][vector[1]] += 1

        coords = np.unravel_index(vector_votacion.argmax(), vector_votacion.shape)

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                img[coords[1] * 10][y] = 0
                img[coords[1] * 10 - 10][y] = 0
                img[x][coords[0] * 10] = 0
                img[x][coords[0] * 10 - 10] = 0

        cv2.imshow(str(i), img)
        cv2.waitKey()

    print(vector_votacion)


def main():
    training_imgs = carga_imagenes_carpeta(CARPETA_TRAIN)
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST)

    tamano_imgane = entrenamiento_orb(training_imgs)
    procesamiento_img(test_imgs, *tamano_imgane)


if __name__ == '__main__':
    main()
