# /usr/bin/python3

import cv2
import time
import os
import numpy as np
from deteccion_haar import *

CARPETA_VIDEOS = "videos/"

out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))


def carga_videos_carpeta(nombre_video):
    video_cap = cv2.VideoCapture(CARPETA_VIDEOS+nombre_video)
    contador_frames = 1

    cascade = cv2.CascadeClassifier(CLASIFICADOR)

    print("Se va a iniciar la carga del vídeo", nombre_video)
    print("###################################################")
    time.sleep(2)

    while True:
        contador_frames += 1
        ret, frame = video_cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coche = cascade.detectMultiScale(gray, 1.1, 1)
            if coche is ():
                print("No se ha encontrado coche")
            for (x, y, w, h) in coche:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Detector de coches', frame)
                out.write(cv2.resize(frame, dsize=(640,480), interpolation=cv2.INTER_CUBIC))
                cv2.waitKey(1)
        else:
            break

    print("FIN")
    video_cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("###################################################")
    print("Numero de frames del vídeo:", contador_frames)


def main():
    carga_videos_carpeta("video2.wmv")


if __name__ == '__main__':
    main()
