# /usr/bin/python3

import cv2
import numpy as np
import os
import time
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
        imagen = cv2.imread(os.path.join(nombre_carpeta, nombre_imagen))
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Se detectan los puntos de interes y se computan los descriptores con ORB
        print("ORB para", i)
        (kps_training, des_training) = orb.detectAndCompute(img, None)
        training_keypoints.append(kps_training)  # se guarda la informacion de cada keypoint
        flann.add([des_training])  # se almacenan los descriptores

    valor_x = list(set(training_img_size_x))[0]  # Este valor se corresponde con la anchura de la imagen de training
    valor_y = list(set(training_img_size_y))[0]  # Este valor se corresponde con la altura de la imagen de training

    print("###################################################")
    print("FIN")
    print()

    return (valor_x, valor_y)  # se devuelve el tamano de la imagen de entrenamiento (en este caso son todas iguales)


def votacion_hough(centro, training_kps, test_kps):
    vector = (centro[0] - training_kps.pt[0], centro[1] - training_kps.pt[1])
    vector = (vector[0] * test_kps.size / training_kps.size, vector[1] * test_kps.size / training_kps.size)

    angulo = np.rad2deg(math.atan2(vector[1], vector[0])) + (test_kps.angle - training_kps.angle)

    modulo = np.sqrt(vector[0] * vector[0] + vector[1] * vector[1])

    vector = (modulo * np.cos(angulo) + test_kps.pt[0], modulo * np.sin(angulo) + test_kps.pt[1])
    vector = (np.uint8(vector[0] / 10), np.uint8(vector[1] / 10))

    return vector


def procesamiento_img_orb(imagen, training_x, training_y):
    orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

    original_size = imagen.shape[::-1]  # tamanio original de la imgane

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    imagen_resized = cv2.resize(imagen, dsize=(training_x, training_y), interpolation=cv2.INTER_CUBIC)

    (kps_test, des_test) = orb.detectAndCompute(imagen_resized, None)
    par = zip(kps_test, des_test)

    # Se crea el vector de votacion a partir del tamano de la imagen reducido por un factor (en este caso 10 que es
    # el que determina el enunciado)
    img_size_y = np.uint8(imagen_resized.shape[0] / 10)
    img_size_x = np.uint8(imagen_resized.shape[1] / 10)

    vector_votacion = np.zeros((img_size_y, img_size_x), dtype=np.uint8)

    for (kp, des) in par:
        # Se obtienen los matches potenciales para cada imagen de test
        matches = flann.knnMatch(des, k=5)
        for vecinos in matches:
            for m in vecinos:
                vector = votacion_hough((225, 110), training_keypoints[m.imgIdx][m.trainIdx], kp)
                if (vector[0] >= 0) & (vector[1] >= 0) & (vector[0] < (img_size_y - 1)) & (
                        vector[1] < (img_size_x - 1)):
                    vector_votacion[vector[0]][vector[1]] += 1

    mejor_voto = vector_votacion.argmax()
    coords = np.unravel_index(mejor_voto, vector_votacion.shape)

    # El centro del circulo viene dado por las coordenadas del mejor vector resultante de la votacion, el radio es un
    # valor arbitrario.
    img_procesada = cv2.circle(imagen_resized, center=(np.uint(coords[0] * 10), np.uint8(coords[1] * 10)), radius=10,
                               color=(0, 255, 0), thickness=2)

    # img_procesada = cv2.cvtColor(img_procesada, cv2.COLOR_GRAY2RGB)

    # img_final = cv2.resize(img_procesada, dsize=original_size, interpolation=cv2.INTER_CUBIC)w
    time.sleep(2)
    return img_procesada


# En caso de que se quiera procesar mas de una imagen se define la siguiente funcion
def detector_coches_orb(test_imgs, training_x, training_y):
    kernel = np.ones((5, 5), np.float32) / 25
    for i, img in enumerate(test_imgs):
        print("=============================================================================")
        print("Para la imagen de test", i)

        dst = cv2.filter2D(img, -1, kernel)
        img_salida = procesamiento_img_orb(dst, training_x, training_y)

        cv2.imshow("Resultado de la imagen", img_salida)
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print("=============================================================================")
    print("FIN")


def main():
    # Carga de imagenes
    training_imgs = carga_imagenes_carpeta(CARPETA_TRAIN)
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST)

    (tr_x, tr_y) = entrenamiento_orb(training_imgs)
    detector_coches_orb(test_imgs, tr_x, tr_y)


if __name__ == '__main__':
    main()
