# /usr/bin/python3

import cv2
import time
import os
import numpy as np

CARPETA_VIDEOS = "videos/"


def carga_videos_carpeta(nombre_video):
    video_cap =cv2.VideoCapture(CARPETA_VIDEOS+nombre_video)
    contador_frames = 1

    print("Se va a iniciar la carga del vídeo", nombre_video)
    print("###################################################")
    time.sleep(2)

    while True:
        contador_frames += 1
        ret, frame = video_cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(nombre_video, gray)
            key = cv2.waitKey(1)
        else:
            break

    print("FIN")
    video_cap.release()
    cv2.destroyAllWindows()

    print("###################################################")
    print("Numero de frames del vídeo:", contador_frames)


def main():
    carga_videos_carpeta("video2.wmv")


if __name__ == '__main__':
    main()
