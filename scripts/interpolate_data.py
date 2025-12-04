"""
Script para interpolar bounding boxes faltantes em uma sequência de frames.

Este arquivo recebe uma lista de detecções (CSV), agrupa por `car_id` e
interpola os bounding boxes do carro e da placa quando existirem lacunas
entre frames consecutivos do mesmo veículo. O objetivo é preencher
frames ausentes para gerar uma sequência contínua de detecções por veículo.

Formato esperado (CSV):
- `frame_nmr`: número do frame (inteiro em string)
- `car_id`: identificador do veículo (pode ser float em string)
- `car_bbox`: bbox do carro no formato "[x1 y1 x2 y2]" (espaços entre valores)
- `license_plate_bbox`: bbox da placa no mesmo formato
- campos opcionais: `license_plate_bbox_score`, `license_number`, `license_number_score`

Saída: uma lista de dicionários com frames interpolados escrita em `test_interpolated.csv`.
"""

import csv
import numpy as np
from scipy.interpolate import interp1d
import os

currentDir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(currentDir)  # sobe 1 nível


def interpolate_bounding_boxes(data):
    # Extrai colunas necessárias dos dados de entrada
    # `frame_numbers`: array de inteiros com os números de frame
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    # `car_ids`: transforma o id (que pode vir como '1.0') em inteiro
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    # Converte os bboxes que vêm como strings "[x1 y1 x2 y2]" em listas de float
    # Observação: o split pressupõe que os valores dentro dos colchetes estão separados por espaços
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    # Itera por cada veículo (car_id) encontrado no CSV
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        # Lista de frames onde esse carro aparece (strings)
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        # Mostra (debug) os frames disponíveis para esse carro
        print(frame_numbers_, car_id)

        # Máscara para filtrar arrays por este car_id
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        # Listas que irão conter os bboxes originais + os interpolados
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        # Primeiro e último frame observados para este veículo
        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        # Percorre cada bbox observado (ordenado pela leitura do CSV)
        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            # Se não for o primeiro elemento, podemos verificar se houve gap
            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                # Valores mais recentes já armazenados nas listas interpoladas
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                # Se houver lacuna de frames (>1), interpolamos linearmente os bboxes
                if frame_number - prev_frame_number > 1:
                    # Quantidade de frames entre prev e atual
                    frames_gap = frame_number - prev_frame_number
                    # Pontos conhecidos para interpolação (x)
                    x = np.array([prev_frame_number, frame_number])
                    # Novos x onde queremos valores (exclui o endpoint para evitar duplicação)
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    # Interpolação linear para coordenadas do carro e da placa
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    # Estendemos as listas com os valores interpolados (ignorando o primeiro, que é prev)
                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            # Adiciona o bbox observado (original)
            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        # Agora convertemos as listas interpoladas em linhas (rows) com frame_nmr sequencial
        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            # Armazena os bboxes como strings com espaços entre valores (mesmo formato de entrada)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            # Se o frame for imputado (não existia originalmente), colocamos valores 0 nos campos relacionados à placa
            if str(frame_number) not in frame_numbers_:
                # Linha imputada: não há leitura da placa -> zeros
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Linha original: tenta recuperar os campos opcionais do CSV original
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

def run_interpolation():
    # Carrega o vídeo
    with open(os.path.join(root, "data", "result.csv"), 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Interpola os dados
    interpolated_data = interpolate_bounding_boxes(data)

    # Atualiza os dados no csv
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    with open(os.path.join(root, "data", "result-interpolated.csv"), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)
        
if __name__ == "__main__":
    run_interpolation()