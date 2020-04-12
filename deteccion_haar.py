# /usr/bin/python3

import cv2
import time
import os
import numpy as np

CARPETA_TEST = 'img/test'
CLASIFICADOR = 'assets/haar/coches.xml'

# Se crea el cascade classifier
cascade = cv2.CascadeClassifier(CLASIFICADOR)

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

def procesamiento_img_haar(imagen):
    # Se convierte la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    coches = cascade.detectMultiScale(gray, 1.1, 1)

    if coches is ():
        print("No se ha encontrado coche")
    for (x, y, w, h) in coches:
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Detector de coches', imagen)
        cv2.waitKey(1)
    #cv2.destroyAllWindows()


def detector_coches(imagenes):
    for i, img in enumerate(imagenes):
        print("PROCESANDO IMAGEN", i)
        procesamiento_img_haar(img)


def main():
    test_imgs = np.array(carga_imagenes_carpeta(CARPETA_TEST))
    detector_coches(test_imgs)


if __name__ == '__main__':
    main()
