# /usr/bin/python3

from deteccion_orb import *
from deteccion_haar import *

CARPETA_VIDEOS = "videos"
CARPETA_SALIDA = "output/videos"


# Sobre como abrir videos y leerlos, se ha basado el codigo en la siguiente web:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

def detector_video_orb(nombre_carpeta):
    tiempos  =[]
    for nombre_video in os.listdir(nombre_carpeta):
        # OPCIONAL: Para guardar la salida en un video
        # out = cv2.VideoWriter(os.path.join(CARPETA_SALIDA, "output_" + str(nombre_video).split('.')[0] + ".avi"),
        # cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

        print("Se va a iniciar la carga del vídeo", nombre_video)
        print("###################################################")
        # time.sleep(2)

        video_cap = cv2.VideoCapture(os.path.join(nombre_carpeta, nombre_video))
        contador_frames = 1

        inicio = time.time()
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if ret:
                #frame_bn = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                imagen_procesada = procesamiento_img_orb(frame, frame.shape[1], frame.shape[0])
                cv2.imshow("Resultado de la imagen", imagen_procesada)
                fin = time.time()

                cv2.waitKey(1)
            else:
                break

        print("FIN")
        fin = time.time()
        print("TIEMPO DE PROCESAMIENTO DEL VÍDEO:", fin-inicio)
        video_cap.release()
        cv2.destroyAllWindows()


def detector_video_haar(nombre_carpeta):
    for nombre_video in os.listdir(nombre_carpeta):
        # OPCIONAL: Para guardar la salida en un video out = cv2.VideoWriter(os.path.join(CARPETA_SALIDA, "output_" +
        # str(nombre_video).split('.')[0] + ".avi"), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

        print("Se va a iniciar la carga del vídeo", nombre_video)
        print("###################################################")
        # time.sleep(2)

        video_cap = cv2.VideoCapture(os.path.join(nombre_carpeta, nombre_video))
        contador_frames = 1
        frames_validos = 1

        inicio = time.time()
        while video_cap.isOpened():
            contador_frames += 1
            ret, frame = video_cap.read()
            if ret:
                frames_validos += 1
                procesamiento_img_haar(frame, contador_frames)
            else:
                break

        print("FIN")
        fin = time.time()
        print("TIEMPO DE PROCESAMIENTO DEL VIDEO", fin-inicio)
        print('FRAMES TOTALES:', contador_frames)
        video_cap.release()
        # out.release()
        cv2.destroyAllWindows()


def main():
    imagenes_test = carga_imagenes_carpeta(CARPETA_TEST)
    entrenamiento_orb(imagenes_test)

    detector_video_haar(CARPETA_VIDEOS)
    detector_video_orb(CARPETA_VIDEOS)


if __name__ == '__main__':
    main()
