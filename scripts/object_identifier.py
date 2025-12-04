"""
Script principal do pipeline.

Fluxo geral:
1. Carrega modelos (YOLO para detecção de veículos + detector de placas).
2. Lê vídeo frame-a-frame.
3. Detecta veículos, filtra classes de interesse e passa as detecções para o
   tracker (Sort) para manter IDs estáveis entre frames.
4. Detecta placas no frame, associa cada placa a um veículo rastreado (se estiver
   contida no bbox do carro) e recorta a placa.
5. Pré-processa o recorte da placa (`preprocess_plate`) e passa para o OCR
   (`read_license_plate`). Se o OCR retornar uma placa válida, armazena no
   dicionário `results` para posterior exportação.

Observações:
- `vehicles` contém os IDs de classes COCO que representam veículos de interesse.
- `results` tem a estrutura {frame_nmr: {car_id: {'car': {...}, 'license_plate': {...}}}}
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
from sort.sort import *
from src.util import get_car, read_license_plate, write_csv
from src.preprocess import preprocess_plate 

def run_object_identifier():
    # Pasta atual
    currentDir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(currentDir)

    # Estrutura que irá guardar os resultados (por frame -> por car_id)
    results = {}

    # Inicializa o tracker SORT (mantém estados e IDs entre frames)
    mot_tracker = Sort()

    # --- Carrega os modelos utilizados ---
    # Modelo COCO para detectar objetos (usado para detectar veículos)
    coco_model = YOLO(os.path.join(root, "models", "yolov11n.pt"))
    # Modelo treinado especificamente para detectar placas
    license_plate_detector = YOLO(os.path.join(root, "models", "license_plate_detector.pt"))

    # --- Abre o vídeo de entrada ---
    cap = cv2.VideoCapture(os.path.join(root, "media", "video.mp4"))

    # Lista de classes COCO que representam veículos que queremos rastrear
    vehicles = [2, 3, 5, 7]

    # Loop principal sobre frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            # Inicializa a chave do frame atual no dicionário de resultados
            results[frame_nmr] = {}

            # --- Detecção de veículos com o modelo COCO ---
            detections = coco_model(frame)[0]
            detections_ = []
            # Converte o formato retornado pelo ultralytics para o formato esperado pelo tracker
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                # Filtra apenas classes de veículos (carros, motos, etc.) conforme `vehicles`
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # --- Atualiza o tracker SORT ---
            # O tracker espera um array Nx5 (x1, y1, x2, y2, score). Se não houver detecções,
            # passa um array vazio com a forma correta.
            if len(detections_) > 0:
                dets_to_update = np.asarray(detections_)
            else:
                dets_to_update = np.empty((0, 5))

            # Atualiza estados do tracker e obtém `track_ids` com formato [[x1,y1,x2,y2,track_id], ...]
            track_ids = mot_tracker.update(dets_to_update)

            # --- Detecção de placas no frame ---
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Tenta associar a placa detectada a algum carro rastreado
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Se achamos um carro correspondente, recorta a placa do frame
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # Pré-processa o recorte para OCR (retorna imagem binarizada)
                    license_plate_crop_thresh = preprocess_plate(license_plate_crop, car_id, frame_nmr)
                    
                    # Lê o texto da placa usando o OCR e normaliza/confia se aplicável
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    # Se OCR retornou uma leitura válida, guarda no dicionário resultados
                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

    # Ao final do processamento de todos os frames, escreve os resultados em CSV
    write_csv(results, os.path.join(root, "data", "result.csv"))

if __name__ == "__main__":
    run_object_identifier()