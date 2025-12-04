"""
Script de visualização para sobrepor placas lidas sobre frames de vídeo.

Este módulo lê o CSV interpolado (`test_interpolated.csv`), captura frames do
vídeo correspondente e desenha caixas, placas recortadas e textos das placas
acima dos veículos. Serve para gerar um vídeo de saída com as placas legíveis
e com um layout harmônico.
"""

import ast
import cv2
import numpy as np
import pandas as pd
import os

currentDir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(currentDir)  # sobe 1 nível

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=5, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def write_video():
    # Carrega resultados interpolados (CSV) gerado pelo pipeline
    results = pd.read_csv(os.path.join(root, "data", "result-interpolated.csv"))

    # --- Abre o vídeo de entrada e prepara o writer de saída ---
    video_path = 'sample.mp4'
    video_path = os.path.join(root, "media", "video.mp4")
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(root, "media", "video-final.mp4"), fourcc, fps, (width, height))


    # Dicionário que armazena o crop da placa redimensionado e o texto final para cada car_id
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        # Seleciona a melhor leitura de placa (maior score) para cada veículo
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])

        # Pega o texto da placa com maior confiança
        lp_text = results[(results['car_id'] == car_id) &
                        (results['license_number_score'] == max_)]['license_number'].iloc[0]

        license_plate[car_id] = {'license_crop': None,
                                'license_plate_number': lp_text}

        # Posiciona o vídeo no frame onde a placa com maior score foi encontrada
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
        ret, frame = cap.read()

        # Extrai o bbox da placa (string -> tupla de floats)
        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # --- AJUSTE HARMÔNICO ---
        # 1. Calculamos o tamanho que o texto vai ocupar na tela (para definir largura do crop)
        (text_width, text_height), _ = cv2.getTextSize(
            lp_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0, # Mesma escala usada lá embaixo
            2)

        # 2. Definimos a largura final da imagem baseada no texto + uma margem (padding)
        # Isso garante que a imagem e o quadrado branco tenham a mesma largura.
        target_width = text_width + 50 # 50px de margem total

        # Evita que fique muito estreito se o texto for curto (ex: erro de leitura "1")
        if target_width < 150:
            target_width = 150

        # 3. Redimensionamos a imagem para essa largura exata e altura fixa (120)
        license_crop = cv2.resize(license_crop,
                                (target_width, 120),
                                interpolation=cv2.INTER_CUBIC)

        # 4. Sharpening (Nitidez) para melhorar aparência ao sobrepor no vídeo
        kernel_sharpening = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
        license_crop = cv2.filter2D(license_crop, -1, kernel_sharpening)

        license_plate[car_id]['license_crop'] = license_crop


    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Leitura frame-a-frame e sobreposição dos elementos
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            # Seleciona as linhas do CSV referentes ao frame atual
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # Desenha borda estilizada do carro
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), thickness=5,
                            line_length_x=50, line_length_y=50)

                # Desenha retângulo vermelho na placa original
                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # Pega a imagem da placa (já redimensionada para o tamanho do texto)
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                H, W, _ = license_crop.shape

                try:
                    # Desenha a placa recortada acima do carro, centralizada horizontalmente
                    frame[int(car_y1) - H - 10 : int(car_y1) - 10,
                        int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    # Fundo branco para o texto (usa a mesma largura W da placa redimensionada)
                    text_bg_height = 60
                    frame[int(car_y1) - H - 10 - text_bg_height : int(car_y1) - H - 10,
                        int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    # Recalcula tamanho do texto apenas para centralizar
                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        2)

                    # Escreve o texto centralizado sobre o fundo branco
                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 10 - (text_bg_height/2) + (text_height/2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0,
                                (0, 0, 0),
                                2)

                except Exception as e:
                    # Se algo falhar (por exemplo índices fora da imagem), apenas ignora essa sobreposição
                    pass

            # Escreve frame no arquivo de saída
            out.write(frame)
            # Opcionalmente redimensiona para visualização mais leve (não altera o arquivo escrito)
            frame = cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()
    
if __name__ == "__main__":
    write_video()