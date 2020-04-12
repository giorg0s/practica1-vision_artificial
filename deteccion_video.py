# /usr/bin/python3

from deteccion_orb import *
from deteccion_haar import *

CARPETA_VIDEOS = "videos"
CARPETA_SALIDA = "output/videos"


def detector_video_orb(nombre_carpeta):
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
                imagen_bn = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                procesamiento_img_orb(imagen_bn, frame.shape[1], frame.shape[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        print("FIN")
        video_cap.release()
        cv2.destroyAllWindows()

        print("###################################################")
        print("Numero de frames del vídeo:", contador_frames)


def detector_video_haar(nombre_carpeta):
    for nombre_video in os.listdir(nombre_carpeta):
        # OPCIONAL: Para guardar la salida en un video out = cv2.VideoWriter(os.path.join(CARPETA_SALIDA, "output_" +
        # str(nombre_video).split('.')[0] + ".avi"), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

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


def main():
    imagenes_test = carga_imagenes_carpeta(CARPETA_TEST)
    entrenamiento_orb(imagenes_test)

    detector_video_haar(CARPETA_VIDEOS)
    detector_video_orb(CARPETA_VIDEOS)


if __name__ == '__main__':
    main()
