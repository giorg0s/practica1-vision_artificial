# /usr/bin/python3

import cv2
import time
import os
import numpy as np
from deteccion_orb import *
from deteccion_haar import *

CARPETA_VIDEOS = "videos"
CARPETA_SALIDA = "output/videos"


def carga_videos_carpeta(nombre_carpeta):
    for nombre_video in os.listdir(nombre_carpeta):
        # OPCIONAL: Para guardar la salida en un video
        # out = cv2.VideoWriter(os.path.join(CARPETA_SALIDA, "output_" + str(nombre_video).split('.')[0] + ".avi"), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

        print("Se va a iniciar la carga del vídeo", nombre_video)
        print("###################################################")
        time.sleep(2)

        video_cap = cv2.VideoCapture(os.path.join(nombre_carpeta, nombre_video))
        contador_frames = 1

        while video_cap.isOpened():
            contador_frames += 1
            ret, frame = video_cap.read()
            if ret:
                procesamiento_img_haar(frame)
            else:
                break

        print("FIN")
        video_cap.release()
        # out.release()
        cv2.destroyAllWindows()

        print("###################################################")
        print("Numero de frames del vídeo:", contador_frames)


def carga_videos_carpeta_orb(nombre_carpeta):
    for nombre_video in os.listdir(nombre_carpeta):
        # OPCIONAL: Para guardar la salida en un video
        # out = cv2.VideoWriter(os.path.join(CARPETA_SALIDA, "output_" + str(nombre_video).split('.')[0] + ".avi"), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

        print("Se va a iniciar la carga del vídeo", nombre_video)
        print("###################################################")
        time.sleep(2)

        video_cap = cv2.VideoCapture(os.path.join(nombre_carpeta, nombre_video))
        contador_frames = 1

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                orb_processing(gray, frame.shape[1], frame.shape[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        print("FIN")
        video_cap.release()
        cv2.destroyAllWindows()

        print("###################################################")
        print("Numero de frames del vídeo:", contador_frames)


def main():
    imagenes_test = carga_imagenes_carpeta(CARPETA_TEST)
    entrenamiento_orb(imagenes_test)

    carga_videos_carpeta(CARPETA_VIDEOS)
    #carga_videos_carpeta_orb(CARPETA_VIDEOS)


if __name__ == '__main__':
    main()
